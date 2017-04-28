#pragma once

#include <memory>
#include <vector>

#include <cstdlib>

#include <backends.hpp>
#include <fvm_multicell.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <cell_group.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <domain_decomposition.hpp>
#include <mc_cell_group.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <sampler_function.hpp>
#include <thread_private_spike_store.hpp>
#include <threading/threading.hpp>
#include <trace_sampler.hpp>
#include <util/nop.hpp>
#include <util/partition.hpp>
#include <util/range.hpp>

namespace nest {
namespace mc {

using gpu_lowered_cell =
    mc_cell_group<fvm::fvm_multicell<gpu::backend>>;

using multicore_lowered_cell =
    mc_cell_group<fvm::fvm_multicell<multicore::backend>>;

class model {
public:
    using communicator_type = communication::communicator<communication::global_policy>;
    using spike_export_function = std::function<void(const std::vector<spike>&)>;

    struct probe_record {
        cell_member_type id;
        probe_spec probe;
    };

    template <typename Iter>
    model(const recipe& rec, backend_policy policy):
        backend_policy_(policy)
    {
        // perform domaint decomposition
        // TODO:

        // set up communicator based on partition
        // TODO: gid_partition
        communicator_ = communicator_type(gid_partition());

        // generate the cell groups in parallel, with one task per cell group
        cell_groups_.resize(domain_.num_local_groups());

        // thread safe vector for constructing the list of probes in parallel
        threading::parallel_vector<probe_record> probe_tmp;

        threading::parallel_for::apply(0, cell_groups_.size(),
            [&](cell_gid_type i) {
                PE("setup", "cells");

                auto gids = domain_.get_group(i);
                std::vector<cell> cells{gids.from-gids.to};

                for (auto gid: util::make_span(gids.from, gids.to)) {
                    auto i = gid-gids.from;
                    cells[i] = rec.get_cell(gid);

                    cell_lid_type j = 0;
                    for (const auto& probe: cells[i].probes()) {
                        cell_member_type probe_id{gid, j++};
                        probe_tmp.push_back({probe_id, probe});
                    }
                }

                if (backend_policy_==backend_policy::use_multicore) {
                    cell_groups_[i] = make_cell_group<multicore_lowered_cell>(gids.from, cells);
                }
                else {
                    cell_groups_[i] = make_cell_group<gpu_lowered_cell>(gids.from, cells);
                }
                PL(2);
            });

        // store probes
        probes_.assign(probe_tmp.begin(), probe_tmp.end());

        // generate the network connections
        for (cell_gid_type i: util::make_span(domain_.first_cell(), domain_.last_cell())) {
            for (const auto& cc: rec.connections_on(i)) {
                connection conn{cc.source, cc.dest, cc.weight, cc.delay};
                communicator_.add_connection(conn);
            }
        }
        communicator_.construct();

        // Allocate an empty queue buffer for each cell group
        // These must be set initially to ensure that a queue is available for each
        // cell group for the first time step.
        current_events().resize(num_groups());
        future_events().resize(num_groups());
    }

    // one cell per group:
    model(const recipe& rec, backend_policy policy):
        model(rec, policy) // TODO
        //model(rec, util::partition_view(util::make_span(0, rec.num_cells()+1)), policy)
    {}

    void reset() {
        t_ = 0.;
        for (auto& group: cell_groups_) {
            group->reset();
        }

        communicator_.reset();

        for(auto& q : current_events()) {
            q.clear();
        }
        for(auto& q : future_events()) {
            q.clear();
        }

        current_spikes().clear();
        previous_spikes().clear();

        util::profilers_restart();
    }

    time_type run(time_type tfinal, time_type dt) {
        // Calculate the size of the largest possible time integration interval
        // before communication of spikes is required.
        // If spike exchange and cell update are serialized, this is the
        // minimum delay of the network, however we use half this period
        // to overlap communication and computation.
        time_type t_interval = communicator_.min_delay()/2;

        time_type tuntil;

        // task that updates cell state in parallel.
        auto update_cells = [&] () {
            threading::parallel_for::apply(
                0u, cell_groups_.size(),
                 [&](unsigned i) {
                    auto &group = cell_groups_[i];

                    PE("stepping","events");
                    group->enqueue_events(current_events()[i]);
                    PL();

                    group->advance(tuntil, dt);

                    PE("events");
                    current_spikes().insert(group->spikes());
                    group->clear_spikes();
                    PL(2);
                });
        };

        // task that performs spike exchange with the spikes generated in
        // the previous integration period, generating the postsynaptic
        // events that must be delivered at the start of the next
        // integration period at the latest.
        auto exchange = [&] () {
            PE("stepping", "communication");

            PE("exchange");
            auto local_spikes = previous_spikes().gather();
            auto global_spikes = communicator_.exchange(local_spikes);
            PL();

            PE("spike output");
            local_export_callback_(local_spikes);
            global_export_callback_(global_spikes.values());
            PL();

            PE("events");
            future_events() = communicator_.make_event_queues(global_spikes);
            PL();

            PL(2);
        };

        while (t_<tfinal) {
            tuntil = std::min(t_+t_interval, tfinal);

            event_queues_.exchange();
            local_spikes_.exchange();

            // empty the spike buffers for the current integration period.
            // these buffers will store the new spikes generated in update_cells.
            current_spikes().clear();

            // run the tasks, overlapping if the threading model and number of
            // available threads permits it.
            threading::task_group g;
            g.run(exchange);
            g.run(update_cells);
            g.wait();

            t_ = tuntil;
        }

        // Run the exchange one last time to ensure that all spikes are output
        // to file.
        event_queues_.exchange();
        local_spikes_.exchange();
        exchange();

        return t_;
    }

    // only thread safe if called outside the run() method
    void add_artificial_spike(cell_member_type source) {
        add_artificial_spike(source, t_);
    }

    // only thread safe if called outside the run() method
    void add_artificial_spike(cell_member_type source, time_type tspike) {
        current_spikes().get().push_back({source, tspike});
    }

    void attach_sampler(cell_member_type probe_id, sampler_function f, time_type tfrom = 0) {
        const auto idx = domain_.local_group_from_gid(probe_id.gid);

        // only attach samplers for local cells
        if (idx) {
            cell_groups_[*idx]->add_sampler(probe_id, f, tfrom);
        }
    }

    const std::vector<probe_record>& probes() const { return probes_; }

    std::size_t num_spikes() const {
        return communicator_.num_spikes();
    }

    std::size_t num_groups() const {
        return cell_groups_.size();
    }

    std::size_t num_cells() const {
        return domain_.num_local_cells();
    }

    // Set event binning policy on all our groups.
    void set_binning_policy(binning_kind policy, time_type bin_interval) {
        for (auto& group: cell_groups_) {
            group->set_binning_policy(policy, bin_interval);
        }
    }

    // access cell_group directly
    cell_group& group(int i) {
        return *cell_groups_[i];
    }

    // register a callback that will perform a export of the global
    // spike vector
    void set_global_spike_callback(spike_export_function export_callback) {
        global_export_callback_ = export_callback;
    }

    // register a callback that will perform a export of the rank local
    // spike vector
    void set_local_spike_callback(spike_export_function export_callback) {
        local_export_callback_ = export_callback;
    }

private:
    domain_decomposition domain_;
    backend_policy backend_policy_;

    time_type t_ = 0.;
    std::vector<std::unique_ptr<cell_group>> cell_groups_;
    communicator_type communicator_;
    std::vector<probe_record> probes_;

    using event_queue_type = typename communicator_type::event_queue;
    util::double_buffer<std::vector<event_queue_type>> event_queues_;

    using local_spike_store_type = thread_private_spike_store;
    util::double_buffer<local_spike_store_type> local_spikes_;

    spike_export_function global_export_callback_ = util::nop_function;
    spike_export_function local_export_callback_ = util::nop_function;

    // Convenience functions that map the spike buffers and event queues onto
    // the appropriate integration interval.
    //
    // To overlap communication and computation, integration intervals of
    // size Delta/2 are used, where Delta is the minimum delay in the global
    // system.
    // From the frame of reference of the current integration period we
    // define three intervals: previous, current and future
    // Then we define the following :
    //      current_spikes : spikes generated in the current interval
    //      previous_spikes: spikes generated in the preceding interval
    //      current_events : events to be delivered at the start of
    //                       the current interval
    //      future_events  : events to be delivered at the start of
    //                       the next interval

    local_spike_store_type& current_spikes()  { return local_spikes_.get(); }
    local_spike_store_type& previous_spikes() { return local_spikes_.other(); }

    std::vector<event_queue_type>& current_events()  { return event_queues_.get(); }
    std::vector<event_queue_type>& future_events()   { return event_queues_.other(); }
};

} // namespace mc
} // namespace nest
