#include <memory>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/export.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>

#include "epoch.hpp"
#include "cell_group.hpp"
#include "cell_group_factory.hpp"
#include "communication/communicator.hpp"
#include "merge_events.hpp"
#include "thread_private_spike_store.hpp"
#include "threading/threading.hpp"
#include "util/maputil.hpp"
#include "util/span.hpp"
#include "profile/profiler_macro.hpp"

namespace arb {

template <typename Seq, typename Value, typename Less = std::less<>>
auto split_sorted_range(Seq&& seq, const Value& v, Less cmp = Less{}) {
    auto canon = util::canonical_view(seq);
    auto it = std::lower_bound(canon.begin(), canon.end(), v, cmp);
    return std::make_pair(
        util::make_range(seq.begin(), it),
        util::make_range(it, seq.end()));
}

// Create a new cell event_lane vector from sorted pending events, previous event_lane events,
// and events from event generators for the given interval.
ARB_ARBOR_API void merge_cell_events(time_type t_from,
                                     time_type t_to,
                                     event_span old_events,
                                     event_span pending,
                                     std::vector<event_generator>& generators,
                                     pse_vector& new_events) {
    PE(communication:enqueue:setup);
    new_events.clear();
    old_events = split_sorted_range(old_events, t_from, event_time_less()).second;
    PL();

    if (!generators.empty()) {
        PE(communication:enqueue:setup);
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

        PE(communication:enqueue:tree);
        merge_events(spanbuf, new_events);
        PL();

        old_events = old_split.second;
        pending = pending_split.second;
    }

    // Merge (remaining) old and pending events.
    PE(communication:enqueue:merge);
    auto n = new_events.size();
    new_events.resize(n+pending.size()+old_events.size());
    std::merge(pending.begin(), pending.end(), old_events.begin(), old_events.end(), new_events.begin()+n);
    PL();
}

class simulation_state {
public:
    simulation_state(const recipe& rec, const domain_decomposition& decomp, context ctx, arb_seed_type seed);

    void update(const recipe& rec);

    void reset();

    time_type run(time_type tfinal, time_type dt);

    sampler_association_handle add_sampler(cell_member_predicate probeset_ids,
                                           schedule sched,
                                           sampler_function f);

    void remove_sampler(sampler_association_handle);

    void remove_all_samplers();

    std::vector<probe_metadata> get_probe_metadata(const cell_address_type&) const;

    std::size_t num_spikes() const {
        return communicator_.num_spikes();
    }

    void set_remote_spike_filter(const spike_predicate& p) { return communicator_.set_remote_spike_filter(p); }

    time_type min_delay() { return communicator_.min_delay(); }

    spike_export_function global_export_callback_;
    spike_export_function local_export_callback_;
    epoch_function epoch_callback_;
    label_resolution_map source_resolution_map_;
    label_resolution_map target_resolution_map_;


    // We do not serialize:
    // - Infrastructure
    //   + ctx: is unserializable due to MPI comm being a handle
    //   + ddc: depends on ctx
    //   + task_system: irrelevant to serdes
    //   + callback/export: unserializable
    //   ===================
    //   These will have to be re-constituted using the recipe/context/load_balance
    //   pathways in the _same_ manner. Otherwise UB ensues.
    //
    // - Internals/Caches
    //   + gid_to_local: will be re-formed via recipe.
    //   + sassoc_handles: ditto
    //   + event_generators: are stateless, will be re-created by recipe.
    //
    // - Data extraction
    //   ===============
    //   Might be required to change.
    //
    // NOTE(TH): We cannot use ARB_SERDES_ENABLE here for two reasons:
    // - cell_groups contains polymorphic pointers.
    // - thread_local_storage cannot be assigned to.
    friend void serialize(serializer& ser, const std::string& k, const simulation_state& t) {
        ARB_SERDES_WRITE(t_interval_);
        ARB_SERDES_WRITE(epoch_);
        ARB_SERDES_WRITE(pending_events_);
        ARB_SERDES_WRITE(event_lanes_);
        ARB_SERDES_WRITE(cell_groups_);
        ser.begin_write_array("local_spikes_");
        serialize(ser, "0", t.local_spikes_[0].gather());
        serialize(ser, "1", t.local_spikes_[1].gather());
        ser.end_write_array();
    }

    friend void deserialize(serializer& ser, const std::string& k, simulation_state& t) {
        ARB_SERDES_READ(t_interval_);
        ARB_SERDES_READ(epoch_);
        ARB_SERDES_READ(pending_events_);
        ARB_SERDES_READ(event_lanes_);
        ARB_SERDES_READ(cell_groups_);
        // custom deserialization to avoid ill-defined copy construction.
        // TODO check whether is OK in actually multi-threaded environments.
        ser.begin_read_array("local_spikes_");
        std::vector<spike> tmp;
        deserialize(ser, "0", tmp);
        t.local_spikes_[0].insert(tmp);
        tmp.clear();
        deserialize(ser, "1", tmp);
        t.local_spikes_[1].insert(tmp);
        ser.end_read_array();
    }

private:
    // Record last computed epoch (integration interval).
    epoch epoch_;

    // Maximum epoch duration.
    time_type t_interval_ = 0;

    std::vector<cell_group_ptr> cell_groups_;

    // One set of event_generators for each local cell
    std::vector<std::vector<event_generator>> event_generators_;

    // Hash table for looking up the the local index of a cell with a given gid
    struct gid_local_info {
        cell_size_type cell_index;
        cell_size_type group_index;
    };
    std::unordered_map<cell_gid_type, gid_local_info> gid_to_local_;

    communicator communicator_;
    context ctx_;
    domain_decomposition ddc_;

    task_system_handle task_system_;

    // Pending events to be delivered.
    std::vector<pse_vector> pending_events_;
    std::array<std::vector<pse_vector>, 2> event_lanes_;

    // Spikes generated by local cell groups.
    std::array<thread_private_spike_store, 2> local_spikes_;

    // Sampler associations handles are managed by a helper class.
    util::handle_set<sampler_association_handle> sassoc_handles_;

    // Accessors to events
    std::vector<pse_vector>& event_lanes(std::ptrdiff_t epoch_id) { return event_lanes_[epoch_id&1]; }
    thread_private_spike_store& local_spikes(std::ptrdiff_t epoch_id) { return local_spikes_[epoch_id&1]; }

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

    // Apply a functional to each local cell in parallel.
    template <typename L>
    void foreach_cell(L&& fn) {
        threading::parallel_for::apply(0, communicator_.num_local_cells(), task_system_.get(), fn);
    }
};

simulation_state::simulation_state(
        const recipe& rec,
        const domain_decomposition& decomp,
        context ctx,
        arb_seed_type seed
    ):
    ctx_{ctx},
    ddc_{decomp},
    task_system_(ctx->thread_pool),
    local_spikes_({thread_private_spike_store(ctx->thread_pool),
                  thread_private_spike_store(ctx->thread_pool)}) {
    // Generate the cell groups in parallel, with one task per cell group.
    auto num_groups = decomp.num_groups();
    cell_groups_.resize(num_groups);
    std::vector<cell_labels_and_gids> cg_sources(num_groups);
    std::vector<cell_labels_and_gids> cg_targets(num_groups);
    foreach_group_index(
        [&](cell_group_ptr& group, int i) {
          PE(init:simulation:group:factory);
          const auto& group_info = decomp.group(i);
          cell_label_range sources, targets;
          auto factory = cell_kind_implementation(group_info.kind, group_info.backend, *ctx_, seed);
          group = factory(group_info.gids, rec, sources, targets);
          PL();
          PE(init:simulation:group:targets_and_sources);
          cg_sources[i] = cell_labels_and_gids(std::move(sources), group_info.gids);
          cg_targets[i] = cell_labels_and_gids(std::move(targets), group_info.gids);
          PL();
        });

    PE(init:simulation:sources);
    cell_labels_and_gids local_sources, local_targets;
    for(const auto& i: util::make_span(num_groups)) {
        local_sources.append(cg_sources.at(i));
        local_targets.append(cg_targets.at(i));
    }
    PL();

    PE(init:simulation:source:MPI);
    auto global_sources = ctx->distributed->gather_cell_labels_and_gids(local_sources);
    PL();

    PE(init:simulation:resolvers);
    source_resolution_map_ = label_resolution_map(std::move(global_sources));
    target_resolution_map_ = label_resolution_map(std::move(local_targets));
    PL();

    PE(init:simulation:comm);
    communicator_ = communicator(rec, ddc_, ctx_);
    PL();
    update(rec);
    epoch_.reset();
}

void simulation_state::update(const recipe& rec) {
    communicator_.update_connections(rec, ddc_, source_resolution_map_, target_resolution_map_);
    // Use half minimum delay of the network for max integration interval.
    t_interval_ = min_delay()/2;

    const auto num_local_cells = communicator_.num_local_cells();
    // Initialize empty buffers for pending events for each local cell
    pending_events_.resize(num_local_cells);
    // Forget old generators, if present
    event_generators_.clear();
    event_generators_.resize(num_local_cells);
    cell_size_type lidx = 0;
    cell_size_type grpidx = 0;
    PE(init:simulation:update:generators);
    auto target_resolution_map_ptr = std::make_shared<label_resolution_map>(target_resolution_map_);
    for (const auto& group_info: ddc_.groups()) {
        for (auto gid: group_info.gids) {
            // Store mapping of gid to local cell index.
            gid_to_local_[gid] = {lidx, grpidx};
            // Resolve event_generator targets; each event generator gets their own resolver state.
            auto event_gens = rec.event_generators(gid);
            for (auto& g: event_gens) {
                g.resolve_label([target_resolution_map_ptr,
                                 event_resolver=resolver(target_resolution_map_ptr.get()),
                                 gid] (const cell_local_label_type& label) mutable {
                        return event_resolver.resolve({gid, label});
                    });
            }
            // Set up the event generators for cell gid.
            event_generators_[lidx] = event_gens;
            ++lidx;
        }
        ++grpidx;
    }
    PL();

    // Create event lane buffers.
    // One buffer is consumed by cell group updates while the other is filled with events for
    // the following epoch. In each buffer there is one lane for each local cell.
    event_lanes_[0].resize(num_local_cells);
    event_lanes_[1].resize(num_local_cells);
}

void simulation_state::reset() {
    epoch_ = epoch();

    // Reset cell group state.
    foreach_group([](cell_group_ptr& group) { group->reset(); });

    // Clear all pending events in the event lanes.
    for (auto& lanes: event_lanes_) {
        for (auto& lane: lanes) lane.clear();
    }

    // Reset all event generators.
    for (auto& lane: event_generators_) {
        for (auto& gen: lane) gen.reset();
    }

    for (auto& lane: pending_events_) lane.clear();
    for (auto& spikes: local_spikes_) spikes.clear();

    communicator_.reset();
    epoch_.reset();
}

time_type simulation_state::run(time_type tfinal, time_type dt) {
    // Progress simulation to time tfinal, through a series of integration epochs
    // of length at most t_interval_. t_interval_ is chosen to be no more than
    // than half the network minimum delay.
    //
    // There are three simulation tasks that can be run partially in parallel:
    //
    // 1. Update:
    //    Ask each cell group to update their state to the end of the integration epoch.
    //    Generated spikes are stored in local_spikes_ for this epoch.
    //
    // 2. Exchange:
    //    Consume local spikes held in local_spikes_ from a previous update, and collect
    //    such spikes from across all ranks.
    //    Translate spikes to local postsynaptic spike events, to be appended to pending_events_.
    //
    // 3. Enqueue events:
    //    Take events from pending_events_, together with any event-generator events for the
    //    next epoch and any left over events from the last epoch, and collate them into
    //    the per-cell event_lanes for the next epoch.
    //
    // Writing U(k) for Update on kth epoch; D(k) for Exchange of spikes generated in the kth epoch;
    // and E(k) for Enqueue of the events required for the kth epoch, there are the following
    // dependencies:
    //
    //     * E(k) precedes U(k).
    //     * U(k) precedes D(k).
    //     * U(k) precedes U(k+1).
    //     * D(k) precedes E(k+2).
    //     * D(k) precedes D(k+1).
    //
    // In the schedule implemented below, U(k) and D(k-1) or U(k) and E(k+1) can be run
    // in parallel, while D and E operations must be serialized (D writes to pending_events_,
    // while E consumes and clears it). The local spike collection and the per-cell event
    // lanes are double buffered.
    //
    // Required state on run() invocation with epoch_.id==k:
    //     * For k≥0,  U(k) and D(k) have completed.
    //
    // Requires state at end of run(), with epoch_.id==k:
    //     * U(k) and D(k) have completed.

    if (!std::isfinite(tfinal) || tfinal < 0) throw std::domain_error("simulation: tfinal must be finite, positive, and in [ms]");
    if (!std::isfinite(dt) || tfinal < 0) throw std::domain_error("simulation: dt must be finite, positive, and in [ms]");

    if (tfinal<=epoch_.t1) return epoch_.t1;

    // Compute following epoch, with max time tfinal.
    auto next_epoch = [tfinal](epoch e, time_type interval) -> epoch {
        epoch next = e;
        next.advance_to(std::min(next.t1+interval, tfinal));
        return next;
    };

    // Update task: advance cell groups to end of current epoch and store spikes in local_spikes_.
    auto update = [this, dt](epoch current) {
        local_spikes(current.id).clear();
        foreach_group_index(
            [&](cell_group_ptr& group, int i) {
                auto queues = util::subrange_view(event_lanes(current.id), communicator_.group_queue_range(i));
                group->advance(current, dt, queues);

                PE(advance:spikes);
                local_spikes(current.id).insert(group->spikes());
                group->clear_spikes();
                PL();
            });
    };

    // Exchange task: gather previous locally generated spikes, distribute across all ranks, and deliver
    // post-synaptic spike events to per-cell pending event vectors.
    auto exchange = [this](epoch prev) {
        // Collate locally generated spikes.
        PE(communication:exchange:gatherlocal);
        auto all_local_spikes = local_spikes(prev.id).gather();
        PL();
        communicator_.remote_ctrl_send_continue(prev);
        // Gather generated spikes across all ranks.
        auto spikes = communicator_.exchange(all_local_spikes);

        // Present spikes to user-supplied callbacks.
        PE(communication:spikeio);
        if (local_export_callback_) local_export_callback_(all_local_spikes);
        if (global_export_callback_) global_export_callback_(spikes.from_local.values());
        PL();

        // Append events formed from global spikes to per-cell pending event queues.
        PE(communication:walkspikes);
        communicator_.make_event_queues(spikes, pending_events_);
        PL();
    };

    // Enqueue task: build event_lanes for next epoch from pending events, event-generator events for the
    // next epoch, and with any unprocessed events from the current event_lanes.
    auto enqueue = [this](epoch next) {
        foreach_cell(
            [&](cell_size_type i) {
                // NOTE Despite the superficial optics, we need to sort by the
                // full key here and _not_ purely by time. With different
                // parallel distributions, the ordering of events with the same
                // time may change. Consider synapses like this
                //
                // NET_RECEIVE (weight) {
                //   if (state < threshold) {
                //      state = state + weight
                //   }
                // }
                //
                // DERIVATIVE dState {
                //   state' = -tau
                // }
                //
                // and we'd end with different behaviours when events with
                // different weights occur at the same time. We also cannot
                // collapse events as with LIF cells by summing weights as this
                // disturbs dynamics in a different way, eg when
                //
                // NET_RECEIVE (weight) {
                //   state = state + 42
                // }
                PE(communication:enqueue:sort);
                util::sort(pending_events_[i]);
                PL();

                event_span pending = util::range_pointer_view(pending_events_[i]);
                event_span old_events = util::range_pointer_view(event_lanes(next.id-1)[i]);

                merge_cell_events(next.t0, next.t1, old_events, pending, event_generators_[i], event_lanes(next.id)[i]);
                pending_events_[i].clear();
            });
    };

    epoch prev = epoch_;
    epoch current = next_epoch(prev, t_interval_);
    epoch next = next_epoch(current, t_interval_);

    if (epoch_callback_) epoch_callback_(current.t0, tfinal);

    if (next.empty()) {
        enqueue(current);
        update(current);
        exchange(current);
        if (epoch_callback_) epoch_callback_(current.t1, tfinal);
    }
    else {
        enqueue(current);
        threading::task_group g(task_system_.get());
        g.run([&]() { enqueue(next); });
        g.run([&]() { update(current); });
        g.wait();
        if (epoch_callback_) epoch_callback_(current.t1, tfinal);

        for (;;) {
            prev = current;
            current = next;
            next = next_epoch(next, t_interval_);
            if (next.empty()) break;

            g.run([&]() { exchange(prev); enqueue(next); });
            g.run([&]() { update(current); });
            g.wait();
            if (epoch_callback_) epoch_callback_(current.t1, tfinal);
        }

        g.run([&]() { exchange(prev); });
        g.run([&]() { update(current); });
        g.wait();

        exchange(current);
        if (epoch_callback_) epoch_callback_(current.t1, tfinal);
    }

    // Record current epoch for next run() invocation.
    epoch_ = current;
    communicator_.remote_ctrl_send_done();
    return current.t1;
}

sampler_association_handle simulation_state::add_sampler(cell_member_predicate probeset_ids,
                                                         schedule sched,
                                                         sampler_function f) {
    sampler_association_handle h = sassoc_handles_.acquire();
    foreach_group(
        [&](cell_group_ptr& group) { group->add_sampler(h, probeset_ids, sched, f); });
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

std::vector<probe_metadata> simulation_state::get_probe_metadata(const cell_address_type& probeset_id) const {
    if (auto linfo = util::value_by_key(gid_to_local_, probeset_id.gid)) {
        return cell_groups_.at(linfo->group_index)->get_probe_metadata(probeset_id);
    }
    else {
        return {};
    }
}

// Simulation class implementations forward to implementation class.

simulation_builder simulation::create(recipe const & rec) { return {rec}; };

simulation::simulation(
    const recipe& rec,
    context ctx,
    const domain_decomposition& decomp,
    arb_seed_type seed)
{
    impl_.reset(new simulation_state(rec, decomp, ctx, seed));
}

void simulation::reset() {
    impl_->reset();
}

void simulation::update(const recipe& rec) { impl_->update(rec); }

time_type simulation::run(const units::quantity& tfinal, const units::quantity& dt) {
    auto dt_ms = dt.value_as(units::ms);
    if (dt_ms <= 0.0 || std::isnan(dt_ms)) throw domain_error("Finite time-step must be supplied.");
    auto tfinal_ms = tfinal.value_as(units::ms);
    if (tfinal_ms <= 0.0 || std::isnan(tfinal_ms)) throw domain_error("Finite time-step must be supplied.");
    return impl_->run(tfinal_ms, dt_ms);
}

sampler_association_handle simulation::add_sampler(
    cell_member_predicate probeset_ids,
    schedule sched,
    sampler_function f)
{
    return impl_->add_sampler(std::move(probeset_ids), std::move(sched), std::move(f));
}

void simulation::remove_sampler(sampler_association_handle h) {
    impl_->remove_sampler(h);
}

void simulation::remove_all_samplers() {
    impl_->remove_all_samplers();
}

std::vector<probe_metadata> simulation::get_probe_metadata(const cell_address_type& probeset_id) const {
    return impl_->get_probe_metadata(probeset_id);
}

std::size_t simulation::num_spikes() const {
    return impl_->num_spikes();
}

void simulation::set_global_spike_callback(spike_export_function export_callback) {
    impl_->global_export_callback_ = std::move(export_callback);
}

void simulation::set_local_spike_callback(spike_export_function export_callback) {
    impl_->local_export_callback_ = std::move(export_callback);
}

void simulation::set_epoch_callback(epoch_function epoch_callback) {
    impl_->epoch_callback_ = std::move(epoch_callback);
}

simulation::simulation(simulation&&) = default;

simulation::~simulation() = default;

time_type simulation::min_delay() { return impl_->min_delay(); }

ARB_ARBOR_API epoch_function epoch_progress_bar() {
    struct impl {
        double t0 = 0;
        bool first = true;

        void operator() (double t, double tfinal) {
            constexpr unsigned bar_width = 50;
            static const std::string bar_buffer(bar_width+1, '-');

            if (first) {
                first = false;
                t0 = t;
            }

            double percentage = (tfinal==t0)? 1: (t-t0)/(tfinal-t0);
            int val = percentage * 100;
            int lpad = percentage * bar_width;
            int rpad = bar_width - lpad;
            printf("\r%3d%% |%.*s%*s|  %12ums", val, lpad, bar_buffer.c_str(), rpad, "", (unsigned)t);

            if (t==tfinal) {
                // Print new line and reset counters on the last step.
                printf("\n");
                t0 = tfinal;
                first = true;
            }
            fflush(stdout);
        }
    };

    return impl{};
}


void serialize(serializer& s, const std::string& k, const simulation& v) {
    serialize(s, k, v.impl_);
}

void deserialize(serializer& s, const std::string& k, simulation& v) {
    deserialize(s, k, v.impl_);
}

// Propagate filters down the stack.
void simulation::set_remote_spike_filter(const spike_predicate& p) { return impl_->set_remote_spike_filter(p); }

} // namespace arb
