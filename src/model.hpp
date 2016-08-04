# pragma once

#include <cstdlib>
#include <vector>

#include <common_types.hpp>
#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_cell.hpp>
#include <recipe.hpp>
#include <thread_private_spike_store.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <profiling/profiler.hpp>

#include "trace_sampler.hpp"

namespace nest {
namespace mc {

template <typename Cell>
class model {
public:
    using cell_group_type = cell_group<Cell>;
    using time_type = typename cell_group_type::time_type;
    using value_type = typename cell_group_type::value_type;
    using communicator_type = communication::communicator<time_type, communication::global_policy>;
    using sampler_function = typename cell_group_type::sampler_function;

    struct probe_record {
        cell_member_type id;
        probe_spec probe;
    };

    model(const recipe& rec, cell_gid_type cell_from, cell_gid_type cell_to):
        cell_from_(cell_from),
        cell_to_(cell_to),
        communicator_(cell_from, cell_to)
    {
        // generate the cell groups in parallel, with one task per cell group
        cell_groups_ = std::vector<cell_group_type>{cell_to_-cell_from_};
        threading::parallel_vector<probe_record> probes;

        threading::parallel_for::apply(cell_from_, cell_to_,
            [&](cell_gid_type i) {
                PE("setup", "cells");
                auto cell = rec.get_cell(i);
                auto idx = i-cell_from_;
                cell_groups_[idx] = cell_group_type(i, cell);

                cell_lid_type j = 0;
                for (const auto& probe: cell.probes()) {
                    cell_member_type probe_id{i,j++};
                    probes.push_back({probe_id, probe});
                }
                PL(2);
            });

        // insert probes
        probes_.assign(probes.begin(), probes.end());

        // generate the network connections
        for (cell_gid_type i=cell_from_; i<cell_to_; ++i) {
            for (const auto& cc: rec.connections_on(i)) {
                connection<time_type> conn{cc.source, cc.dest, cc.weight, cc.delay};
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

    void reset() {
        t_ = 0.;
        for (auto& group: cell_groups_) {
            group.reset();
        }
        communicator_.reset();
    }

    time_type run(time_type tfinal, time_type dt) {
        time_type min_delay = communicator_.min_delay()/2;
        while (t_<tfinal) {
            auto tuntil = std::min(t_+min_delay, tfinal);

            event_queues_.exchange();
            local_spikes_.exchange();

            // empty the spike buffers for the current integration period.
            // these buffers will store the new spikes generated in update_cells.
            current_spikes().clear();

            // TODO this needs a threading wrapper
            tbb::task_group g;

            auto update_cells = [&] () {
                threading::parallel_for::apply(
                    0u, cell_groups_.size(),
                    [&](unsigned i) {
                        auto &group = cell_groups_[i];

                        PE("stepping","events");
                        group.enqueue_events(current_events()[i]);
                        PL();

                        group.advance(tuntil, dt);

                        PE("events");
                        current_spikes().insert(group.spikes());
                        group.clear_spikes();
                        PL(2);
                    });
            };

            auto exchange = [&] () {
                PE("stepping", "exchange");
                auto local_spikes = previous_spikes().gather();
                future_events() = communicator_.exchange(local_spikes);
                PL(2);
            };

            g.run(exchange);
            g.run(update_cells);
            g.wait();

            t_ = tuntil;
        }
        return t_;
    }

    // TODO : these two calls are only thread safe if called outside the main time
    // stepping loop.
    void add_artificial_spike(cell_member_type source) {
        add_artificial_spike(source, t_);
    }

    void add_artificial_spike(cell_member_type source, time_type tspike) {
        current_spikes().get().push_back({source, tspike});
    }

    void attach_sampler(cell_member_type probe_id, sampler_function f, time_type tfrom = 0) {
        // TODO: translate probe_id.gid to appropriate group, but for now 1-1.
        if (probe_id.gid<cell_from_ || probe_id.gid>=cell_to_) {
            return;
        }
        cell_groups_[probe_id.gid-cell_from_].add_sampler(probe_id, f, tfrom);
    }

    const std::vector<probe_record>& probes() const { return probes_; }

    std::size_t num_spikes() const { return communicator_.num_spikes(); }
    std::size_t num_groups() const { return cell_groups_.size(); }

private:
    cell_gid_type cell_from_;
    cell_gid_type cell_to_;
    time_type t_ = 0.;
    std::vector<cell_group_type> cell_groups_;
    communicator_type communicator_;
    std::vector<probe_record> probes_;
    using spike_type = typename communicator_type::spike_type;

    using event_queue_type = typename communicator_type::event_queue;
    util::double_buffer< std::vector<event_queue_type> > event_queues_;

    using local_spike_store_type = thread_private_spike_store<time_type>;
    util::double_buffer< local_spike_store_type > local_spikes_;

    local_spike_store_type& current_spikes()  { return local_spikes_.get(); }
    local_spike_store_type& previous_spikes() { return local_spikes_.other(); }
    std::vector<event_queue_type>& current_events()  { return event_queues_.get(); }
    std::vector<event_queue_type>& future_events()   { return event_queues_.other(); }
};

} // namespace mc
} // namespace nest
