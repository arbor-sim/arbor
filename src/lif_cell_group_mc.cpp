#include <lif_cell_group_mc.hpp>

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group_mc::lif_cell_group_mc(std::vector<cell_gid_type> gids, const recipe& rec):
gids_(std::move(gids))
{
    // Default to no binning of events
    set_binning_policy(binning_kind::none, 0);

    // reserve
    cells_.reserve(gids_.size());

    // resize
    last_time_updated_.resize(gids_.size());

    for (auto lid: util::make_span(0, gids_.size())) {
        cells_.push_back(util::any_cast<lif_cell_description>(rec.get_cell_description(gids_[lid])));
    }
}

cell_kind lif_cell_group_mc::get_cell_kind() const {
    return cell_kind::lif_neuron;
}

void lif_cell_group_mc::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE("lif");
    if (event_lanes.size() > 0) {
        for (auto lid: util::make_span(0, gids_.size())) {
            // std::cout << "Received " << event_lanes[lid].size() << std::endl;
            // Advance each cell independently.
            advance_cell(ep.tfinal, dt, lid, event_lanes[lid]);
        }
    }
    PL();
}

const std::vector<spike>& lif_cell_group_mc::spikes() const {
    return spikes_;
}

void lif_cell_group_mc::clear_spikes() {
    spikes_.clear();
}

// TODO: implement sampler
void lif_cell_group_mc::add_sampler(sampler_association_handle h, cell_member_predicate probe_ids,
                                    schedule sched, sampler_function fn, sampling_policy policy) {}
void lif_cell_group_mc::remove_sampler(sampler_association_handle h) {}
void lif_cell_group_mc::remove_all_samplers() {}

// TODO: implement binner_
void lif_cell_group_mc::set_binning_policy(binning_kind policy, time_type bin_interval) {
}

void lif_cell_group_mc::reset() {
    spikes_.clear();
    last_time_updated_.clear();
}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group_mc::advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, pse_vector& event_lane) {
    // Current time of last update.
    auto t = last_time_updated_[lid];
    auto& cell = cells_[lid];
    unsigned i = 0;

    // If a neuron was in the refractory period,
    // ignore any new events that happened before t,
    // including poisson events as well.
    for (auto ev : event_lane) {
        if (ev.time >= t) break;
        i++;
    }

    // Integrate until tfinal using the exact solution of membrane voltage differential equation.
    for (; i < event_lane.size(); i++) {
        auto ev = event_lane[i];
        if (ev.time >= tfinal) break;

        auto weight = ev.weight;
        auto time = ev.time;

        // If a neuron is in refractory period, ignore this event.
        if (time < t) continue;

        // if there are events that happened at the same time as this event, process them as well
        while (i + 1 < event_lane.size() && event_lane[i+1].time <= time) {
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

            // Advance last_time_updated.
            t += cell.t_ref;

            // Reset the voltage to resting potential.
            cell.V_m = cell.E_L;
        }
        // This is the last time a cell was updated.
        last_time_updated_[lid] = t;
    }
}
