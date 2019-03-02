#include <lif_cell_group.hpp>

#include "profile/profiler_macro.hpp"
#include "util/span.hpp"

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group::lif_cell_group(const std::vector<cell_gid_type>& gids, const recipe& rec):
    gids_(gids)
{
    // Default to no binning of events
    set_binning_policy(binning_kind::none, 0);

    cells_.reserve(gids_.size());
    last_time_updated_.resize(gids_.size());

    for (auto lid: util::make_span(gids_.size())) {
        cells_.push_back(util::any_cast<lif_cell>(rec.get_cell_description(gids_[lid])));
    }
}

cell_kind lif_cell_group::get_cell_kind() const {
    return cell_kind::lif;
}

void lif_cell_group::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE(advance_lif);
    if (event_lanes.size() > 0) {
        for (auto lid: util::make_span(gids_.size())) {
            // Advance each cell independently.
            advance_cell(ep.tfinal, dt, lid, event_lanes[lid]);
        }
    }
    PL();
}

const std::vector<spike>& lif_cell_group::spikes() const {
    return spikes_;
}

void lif_cell_group::clear_spikes() {
    spikes_.clear();
}

// TODO: implement sampler
void lif_cell_group::add_sampler(sampler_association_handle h, cell_member_predicate probe_ids,
                                    schedule sched, sampler_function fn, sampling_policy policy) {}
void lif_cell_group::remove_sampler(sampler_association_handle h) {}
void lif_cell_group::remove_all_samplers() {}

// TODO: implement binner_
void lif_cell_group::set_binning_policy(binning_kind policy, time_type bin_interval) {
}

void lif_cell_group::reset() {
    spikes_.clear();
    last_time_updated_.clear();
}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group::advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, pse_vector& event_lane) {
    // Current time of last update.
    auto t = last_time_updated_[lid];
    auto& cell = cells_[lid];
    const auto n_events = event_lane.size();

    // Integrate until tfinal using the exact solution of membrane voltage differential equation.
    for (unsigned i=0; i<n_events; ++i ) {
        auto& ev = event_lane[i];
        const auto time = ev.time;
        auto weight = ev.weight;

        if (time < t) continue;        // skip event if a neuron is in refactory period
        if (time >= tfinal) break;     // end of integration interval

        // if there are events that happened at the same time as this event, process them as well
        while (i + 1 < n_events && event_lane[i+1].time <= time) {
            weight += event_lane[i+1].weight;
            i++;
        }

        // Let the membrane potential decay.
        auto decay = exp(-(time - t) / cell.tau_m);
        cell.V_m *= decay;
        auto update = weight / cell.C_m;
        // Add jump due to spike.
        cell.V_m += update;
        t = time;
        // If crossing threshold occurred
        if (cell.V_m >= cell.V_th) {
            cell_member_type spike_neuron_gid = {gids_[lid], 0};
            spike s = {spike_neuron_gid, t};
            spikes_.push_back(s);

            // Advance the last_time_updated to account for the refractory period.
            t += cell.t_ref;

            // Reset the voltage to resting potential.
            cell.V_m = cell.E_L;
        }
    }

    // This is the last time a cell was updated.
    last_time_updated_[lid] = t;
}
