#include <memory>
#include <set>
#include <vector>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>

#include "cell_group.hpp"
#include "cell_group_factory.hpp"
#include "communication/communicator.hpp"
#include "execution_context.hpp"
#include "merge_events.hpp"
#include "thread_private_spike_store.hpp"
#include "threading/threading.hpp"
#include "util/double_buffer.hpp"
#include "util/filter.hpp"
#include "util/maputil.hpp"
#include "util/partition.hpp"
#include "util/span.hpp"
#include "profile/profiler_macro.hpp"

namespace arb {

class spike_double_buffer {
    util::double_buffer<thread_private_spike_store> buffer_;

public:
    // Convenience functions that map the spike buffers onto the appropriate
    // integration interval.
    //
    // To overlap communication and computation, integration intervals of
    // size Delta/2 are used, where Delta is the minimum delay in the global
    // system.
    // From the frame of reference of the current integration period we
    // define three intervals: previous, current and future
    // Then we define the following :
    //      current:  spikes generated in the current interval
    //      previous: spikes generated in the preceding interval

    spike_double_buffer(thread_private_spike_store l, thread_private_spike_store r):
            buffer_(std::move(l), std::move(r)) {}

    thread_private_spike_store& current()  { return buffer_.get(); }
    thread_private_spike_store& previous() { return buffer_.other(); }
    void exchange() { buffer_.exchange(); }
};

class simulation_state {
public:
    simulation_state(const recipe& rec, const domain_decomposition& decomp, execution_context ctx);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    sampler_association_handle add_sampler(cell_member_predicate probe_ids,
        schedule sched, sampler_function f, sampling_policy policy = sampling_policy::lax);

    void remove_sampler(sampler_association_handle);

    void remove_all_samplers();

    std::size_t num_spikes() const {
        return communicator_.num_spikes();
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval);

    void inject_events(const pse_vector& events);

    spike_export_function global_export_callback_;
    spike_export_function local_export_callback_;

private:
    // Private helper function that sets up the event lanes for an epoch.
    // See comments on implementation for more information.
    void setup_events(time_type t_from, time_type time_to, std::size_t epoch_id);

    std::vector<pse_vector>& event_lanes(std::size_t epoch_id) {
        return event_lanes_[epoch_id%2];
    }

    // keep track of information about the current integration interval
    epoch epoch_;

    time_type t_ = 0.;
    time_type min_delay_;
    std::vector<cell_group_ptr> cell_groups_;

    // one set of event_generators for each local cell
    std::vector<std::vector<event_generator>> event_generators_;

    std::unique_ptr<spike_double_buffer> local_spikes_;

    // Hash table for looking up the the local index of a cell with a given gid
    std::unordered_map<cell_gid_type, cell_size_type> gid_to_local_;

    communicator communicator_;

    task_system_handle task_system_;

    // Pending events to be delivered.
    std::array<std::vector<pse_vector>, 2> event_lanes_;
    std::vector<pse_vector> pending_events_;

    // Sampler associations handles are managed by a helper class.
    util::handle_set<sampler_association_handle> sassoc_handles_;

    // Apply a functional to each cell group in parallel.
    template <typename L>
    void foreach_group(L&& fn) {
        threading::parallel_for::apply(0, cell_groups_.size(), task_system_.get(),
            [&, fn = std::forward<L>(fn)](int i) { fn(cell_groups_[i]); });
    }

    // Apply a functional to each cell group in parallel, supplying
    // the cell group pointer reference and index.
    template <typename L>
    void foreach_group_index(L&& fn) {
        threading::parallel_for::apply(0, cell_groups_.size(), task_system_.get(),
            [&, fn = std::forward<L>(fn)](int i) { fn(cell_groups_[i], i); });
    }
};

simulation_state::simulation_state(
        const recipe& rec,
        const domain_decomposition& decomp,
        execution_context ctx
    ):
    local_spikes_(new spike_double_buffer(thread_private_spike_store(ctx.thread_pool),
                                          thread_private_spike_store(ctx.thread_pool))),
    communicator_(rec, decomp, ctx),
    task_system_(ctx.thread_pool)
{
    const auto num_local_cells = communicator_.num_local_cells();

    // Cache the minimum delay of the network
    min_delay_ = communicator_.min_delay();

    // Initialize empty buffers for pending events for each local cell
    pending_events_.resize(num_local_cells);

    event_generators_.resize(num_local_cells);
    cell_local_size_type lidx = 0;
    for (const auto& group_info: decomp.groups) {
        for (auto gid: group_info.gids) {
            // Store mapping of gid to local cell index.
            gid_to_local_[gid] = lidx;

            // Set up the event generators for cell gid.
            event_generators_[lidx] = rec.event_generators(gid);
            ++lidx;
        }
    }

    // Generate the cell groups in parallel, with one task per cell group.
    cell_groups_.resize(decomp.groups.size());
    foreach_group_index(
        [&](cell_group_ptr& group, int i) {
            const auto& group_info = decomp.groups[i];
            auto factory = cell_kind_implementation(group_info.kind, group_info.backend, ctx);
            group = factory(group_info.gids, rec);
        });

    // Create event lane buffers.
    // There is one set for each epoch: current (0) and next (1).
    // For each epoch there is one lane for each cell in the cell group.
    event_lanes_[0].resize(num_local_cells);
    event_lanes_[1].resize(num_local_cells);
}

void simulation_state::reset() {
    t_ = 0.;

    // Reset cell group state.
    foreach_group(
        [](cell_group_ptr& group) { group->reset(); });

    // Clear all pending events in the event lanes.
    for (auto& lanes: event_lanes_) {
        for (auto& lane: lanes) {
            lane.clear();
        }
    }

    // Reset all event generators.
    for (auto& lane: event_generators_) {
        for (auto& gen: lane) {
            gen.reset();
        }
    }

    for (auto& lane: pending_events_) {
        lane.clear();
    }

    communicator_.reset();

    local_spikes_->current().clear();
    local_spikes_->previous().clear();
}

time_type simulation_state::run(time_type tfinal, time_type dt) {
    // Calculate the size of the largest possible time integration interval
    // before communication of spikes is required.
    // If spike exchange and cell update are serialized, this is the
    // minimum delay of the network, however we use half this period
    // to overlap communication and computation.
    const time_type t_interval = min_delay_/2;

    // task that updates cell state in parallel.
    auto update_cells = [&] () {
        foreach_group_index(
            [&](cell_group_ptr& group, int i) {
                auto queues = util::subrange_view(event_lanes(epoch_.id), communicator_.group_queue_range(i));
                group->advance(epoch_, dt, queues);

                PE(advance_spikes);
                local_spikes_->current().insert(group->spikes());
                group->clear_spikes();
                PL();
            });
    };

    // task that performs spike exchange with the spikes generated in
    // the previous integration period, generating the postsynaptic
    // events that must be delivered at the start of the next
    // integration period at the latest.
    auto exchange = [&] () {
        PE(communication_exchange_gatherlocal);
        auto local_spikes = local_spikes_->previous().gather();
        PL();
        auto global_spikes = communicator_.exchange(local_spikes);

        PE(communication_spikeio);
        if (local_export_callback_) {
            local_export_callback_(local_spikes);
        }
        if (global_export_callback_) {
            global_export_callback_(global_spikes.values());
        }
        PL();

        PE(communication_walkspikes);
        communicator_.make_event_queues(global_spikes, pending_events_);
        PL();

        const auto t0 = epoch_.tfinal;
        const auto t1 = std::min(tfinal, t0+t_interval);
        setup_events(t0, t1, epoch_.id);
    };

    time_type tuntil = std::min(t_+t_interval, tfinal);
    epoch_ = epoch(0, tuntil);
    setup_events(t_, tuntil, 1);
    while (t_<tfinal) {
        local_spikes_->exchange();

        // empty the spike buffers for the current integration period.
        // these buffers will store the new spikes generated in update_cells.
        local_spikes_->current().clear();

        // run the tasks, overlapping if the threading model and number of
        // available threads permits it.
        threading::task_group g(task_system_.get());
        g.run(exchange);
        g.run(update_cells);
        g.wait();

        t_ = tuntil;

        tuntil = std::min(t_+t_interval, tfinal);
        epoch_.advance(tuntil);
    }

    // Run the exchange one last time to ensure that all spikes are output to file.
    local_spikes_->exchange();
    exchange();

    return t_;
}

template <typename Seq, typename Value, typename Less = std::less<>>
auto split_sorted_range(Seq&& seq, const Value& v, Less cmp = Less{}) {
    auto canon = util::canonical_view(seq);
    auto it = std::lower_bound(canon.begin(), canon.end(), v, cmp);
    return std::make_pair(
        util::make_range(seq.begin(), it),
        util::make_range(it, seq.end()));
}

// Populate the event lanes for epoch+1 (i.e event_lanes_[epoch+1)]
// Update each lane in parallel, if supported by the threading backend.
// On completion event_lanes[epoch+1] will contain sorted lists of events with
// delivery times due in or after epoch+1. The events will be taken from the
// following sources:
//      event_lanes[epoch]: take all events ≥ t_from
//      event_generators  : take all events < t_to
//      pending_events    : take all events

// merge_cell_events() is a separate function for unit testing purposes.
void merge_cell_events(
    time_type t_from,
    time_type t_to,
    event_span old_events,
    event_span pending,
    std::vector<event_generator>& generators,
    pse_vector& new_events)
{
    PE(communication_enqueue_setup);
    new_events.clear();
    old_events = split_sorted_range(old_events, t_from, event_time_less()).second;
    PL();

    if (!generators.empty()) {
        PE(communication_enqueue_setup);
        // Tree-merge events in [t_from, t_to) from old, pending and generator events.

        std::vector<event_span> spanbuf;
        spanbuf.reserve(2+generators.size());

        auto old_split = split_sorted_range(old_events, t_to, event_time_less());
        auto pending_split = split_sorted_range(pending, t_to, event_time_less());

        spanbuf.push_back(old_split.first);
        spanbuf.push_back(pending_split.first);

        for (auto& g: generators) {
            event_span evs = g.events(t_from, t_to);
            if (!evs.empty()) {
                spanbuf.push_back(evs);
            }
        }
        PL();

        PE(communication_enqueue_tree);
        tree_merge_events(spanbuf, new_events);
        PL();

        old_events = old_split.second;
        pending = pending_split.second;
    }

    // Merge (remaining) old and pending events.
    PE(communication_enqueue_merge);
    auto n = new_events.size();
    new_events.resize(n+pending.size()+old_events.size());
    std::merge(pending.begin(), pending.end(), old_events.begin(), old_events.end(), new_events.begin()+n);
    PL();
}

void simulation_state::setup_events(time_type t_from, time_type t_to, std::size_t epoch) {
    const auto n = communicator_.num_local_cells();
    threading::parallel_for::apply(0, n, task_system_.get(),
        [&](cell_size_type i) {
            PE(communication_enqueue_sort);
            util::sort(pending_events_[i]);
            PL();

            event_span pending = util::range_pointer_view(pending_events_[i]);
            event_span old_events = util::range_pointer_view(event_lanes(epoch)[i]);

            merge_cell_events(t_from, t_to, old_events, pending, event_generators_[i], event_lanes(epoch+1)[i]);
            pending_events_[i].clear();
            });
}

sampler_association_handle simulation_state::add_sampler(
        cell_member_predicate probe_ids,
        schedule sched,
        sampler_function f,
        sampling_policy policy)
{
    sampler_association_handle h = sassoc_handles_.acquire();

    foreach_group(
        [&](cell_group_ptr& group) { group->add_sampler(h, probe_ids, sched, f, policy); });

    return h;
}

void simulation_state::remove_sampler(sampler_association_handle h) {
    foreach_group(
        [h](cell_group_ptr& group) { group->remove_sampler(h); });

    sassoc_handles_.release(h);
}

void simulation_state::remove_all_samplers() {
    foreach_group(
        [](cell_group_ptr& group) { group->remove_all_samplers(); });

    sassoc_handles_.clear();
}

void simulation_state::set_binning_policy(binning_kind policy, time_type bin_interval) {
    foreach_group(
        [&](cell_group_ptr& group) { group->set_binning_policy(policy, bin_interval); });
}

void simulation_state::inject_events(const pse_vector& events) {
    // Push all events that are to be delivered to local cells into the
    // pending event list for the event's target cell.
    for (auto& e: events) {
        if (e.time<t_) {
            throw bad_event_time(e.time, t_);
        }
        // gid_to_local_ maps gid to index into local set of cells.
        if (auto lidx = util::value_by_key(gid_to_local_, e.target.gid)) {
            pending_events_[*lidx].push_back(e);
        }
    }
}

// Simulation class implementations forward to implementation class.

simulation::simulation(
    const recipe& rec,
    const domain_decomposition& decomp,
    const context& ctx)
{
    impl_.reset(new simulation_state(rec, decomp, *ctx));
}

void simulation::reset() {
    impl_->reset();
}

time_type simulation::run(time_type tfinal, time_type dt) {
    return impl_->run(tfinal, dt);
}

sampler_association_handle simulation::add_sampler(
    cell_member_predicate probe_ids,
    schedule sched,
    sampler_function f,
    sampling_policy policy)
{
    return impl_->add_sampler(std::move(probe_ids), std::move(sched), std::move(f), policy);
}

void simulation::remove_sampler(sampler_association_handle h) {
    impl_->remove_sampler(h);
}

void simulation::remove_all_samplers() {
    impl_->remove_all_samplers();
}

std::size_t simulation::num_spikes() const {
    return impl_->num_spikes();
}

void simulation::set_binning_policy(binning_kind policy, time_type bin_interval) {
    impl_->set_binning_policy(policy, bin_interval);
}

void simulation::set_global_spike_callback(spike_export_function export_callback) {
    impl_->global_export_callback_ = std::move(export_callback);
}

void simulation::set_local_spike_callback(spike_export_function export_callback) {
    impl_->local_export_callback_ = std::move(export_callback);
}

void simulation::inject_events(const pse_vector& events) {
    impl_->inject_events(events);
}

simulation::~simulation() = default;

} // namespace arb
