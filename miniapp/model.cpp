#include <cstdlib>
#include <vector>

#include <catypes.hpp>
#include <cell.hpp>
#include <cell_group.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <fvm_cell.hpp>
#include <profiling/profiler.hpp>
#include <recipe.hpp>
#include <threading/threading.hpp>

#include "model.hpp"
#include "trace_sampler.hpp"

namespace nest {
namespace mc {

model::model(const recipe &rec, cell_gid_type cell_from, cell_gid_type cell_to, time_type sample_dt) {
    // construct cell groups (one cell per group) and attach samplers
    cell_groups_ = std::vector<cell_group_type>{cell_to-cell_from};
    samplers_.resize(cell_to-cell_from);

    threading::parallel_for::apply(cell_from, cell_to, 
        [&](cell_gid_type i) {
            PE("setup", "cells");
            auto cell = rec.get_cell(i);
            auto idx = i-cell_from;
            cell_groups_[idx] = cell_group_type(i, cell);

            cell_local_index_type j = 0;
            for (const auto& probe: cell.probes()) {
                cell_member_type probe_id{i,j++};
                samplers_[idx].emplace_back(probe_id, probe.kind, probe.location, sample_dt);
                const auto &sampler = samplers_[idx].back();
                cell_groups_[idx].add_sampler(probe_id, sampler, sampler.next_sample_t());
            }
            PL(2);
        });

    // initialise communicator
    communicator_ = communicator_type(cell_from, cell_to);
}

void model::reset() {
    t_ = 0.;
    // otherwise unimplemented
    std::abort();
}

model::time_type model::run(time_type tfinal, time_type dt) {
    time_type min_delay = communicator_.min_delay();
    while (t_<tfinal) {
        auto tuntil = std::min(t_+min_delay, tfinal);
        threading::parallel_for::apply(
            0u, cell_groups_.size(),
            [&](unsigned i) {
                auto &group = cell_groups_[i];

                PE("stepping","events");
                group.enqueue_events(communicator_.queue(i));
                PL();

                group.advance(tuntil, dt);

                PE("events");
                communicator_.add_spikes(group.spikes());
                group.clear_spikes();
                PL(2);
            });

        PE("stepping", "exchange");
        communicator_.exchange();
        PL(2);

        t_ = tuntil;
    }
    return t_;
}

void model::write_traces() const {
    for (auto& gid_samplers: samplers_) {
        for (auto& s: gid_samplers) {
            s.write_trace();
        }
    }
}

void model::add_artificial_spike(cell_member_type source, time_type tspike) {
    communicator_.add_spike({source, tspike});
}

} // namespace mc
} // namespace nest
