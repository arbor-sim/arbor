#include <lif_cell_group_mc.hpp>

using namespace nest::mc;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group_mc::lif_cell_group_mc(cell_gid_type first_gid, const std::vector<util::unique_any>& cells):
gid_base_(first_gid)
{
    cells_.reserve(cells.size());
    lambda_.resize(cells.size());

    generator_.resize(cells.size());
    next_poiss_time_.resize(cells.size());

    cell_events_.resize(cells.size());
    last_time_updated_.resize(cells.size());

    for (const auto& cell : cells) {
        cells_.push_back(util::any_cast<lif_cell_description>(cell));
    }

    // Initialize variables for the external Poisson input.
    for (auto lid : util::make_span(0, cells_.size())) {
        EXPECTS(cells_[lid].n_poiss >= 0);
        EXPECTS(cells_[lid].w_poiss >= 0);
        EXPECTS(cells_[lid].d_poiss >= 0);
        EXPECTS(cells_[lid].rate >= 0);

        // If a cell receives some external Poisson input then initialize the corresponding variables.
        if (cells_[lid].n_poiss > 0) {
            lambda_[lid] = (1.0/(cells_[lid].rate * cells_[lid].n_poiss));
            generator_[lid].seed(1000 + first_gid + lid);
            sample_next_poisson(lid);
        }
    }
}

cell_kind lif_cell_group_mc::get_cell_kind() const {
    return cell_kind::lif_neuron;
}

void lif_cell_group_mc::advance(time_type tfinal, time_type dt) {
    PE("lif");
    for (size_t lid = 0; lid < cells_.size(); ++lid) {
        // Advance each cell independently.
        advance_cell(tfinal, dt, lid);
    }
    PL();
}

void lif_cell_group_mc::enqueue_events(const std::vector<postsynaptic_spike_event>& events) {
    // Distribute incoming events to individual cells.
    for (auto& e: events) {
        cell_events_[e.target.gid - gid_base_].push(e);
    }
}

const std::vector<spike>& lif_cell_group_mc::spikes() const {
    return spikes_;
}

void lif_cell_group_mc::clear_spikes() {
    spikes_.clear();
}

// TODO: implement sampler
void lif_cell_group_mc::add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time) {}

// TODO: implement binner_
void lif_cell_group_mc::set_binning_policy(binning_kind policy, time_type bin_interval) {
}

// no probes in single-compartment cells
std::vector<probe_record> lif_cell_group_mc::probes() const {
    return {};
}

void lif_cell_group_mc::reset() {
    spikes_.clear();
    // Clear all the event queues.
    for (auto& queue : cell_events_) {
        queue.clear();
    }
}

// Samples next poisson spike.
void lif_cell_group_mc::sample_next_poisson(cell_gid_type lid) {
    next_poiss_time_[lid] += exp_dist_(generator_[lid]) * lambda_[lid];
}

// Returns the time of the next poisson event for given neuron,
// taking into accout the delay of poisson spikes,
// without sampling a new Poisson event time.
time_type lif_cell_group_mc::next_poisson_event(cell_gid_type lid) {
    return next_poiss_time_[lid] + cells_[lid].d_poiss;
}

// Returns the next most recent event that is yet to be processed.
// It can be either Poisson event or the queue event.
// Only events that happened before tfinal are considered.
util::optional<postsynaptic_spike_event> lif_cell_group_mc::next_event(cell_gid_type lid, time_type tfinal) {
    auto t_poiss = next_poisson_event(lid);

    // t_queue < {t_poiss, tfinal} => return t_queue
    if (auto ev = cell_events_[lid].pop_if_before(std::min(tfinal, t_poiss))) {
        return ev;
    }

    if (t_poiss < tfinal) {
        // t_queue < t_poiss < tfinal => return t_queue
        if (auto ev = cell_events_[lid].pop_if_before(t_poiss)) {
            return ev;
        }
        // t_poiss < {t_queue, tfinal} => return t_poiss
        sample_next_poisson(lid);
        return postsynaptic_spike_event{{cell_lid_type(gid_base_ + lid), 0}, t_poiss, cells_[lid].w_poiss};
    }

    // t_queue < tfinal < t_poiss => return t_queue
    return cell_events_[lid].pop_if_before(tfinal);

}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group_mc::advance_cell(time_type tfinal, time_type dt, cell_gid_type lid) {
    // Current time of last update.
    auto t = last_time_updated_[lid];
    auto& cell = cells_[lid];

    // If a neuron was in the refractory period,
    // ignore any new events that happened before t,
    // including poisson events as well.
    while (auto ev = next_event(lid, t)) {};

    // Integrate until tfinal using the exact solution of membrane voltage differential equation.
    while (auto ev = next_event(lid, tfinal)) {
        auto weight = ev->weight;
        auto event_time = ev->time;

        // If a neuron is in refractory period, ignore this event.
        if (event_time < t) {
            continue;
        }

        // Let the membrane potential decay.
        cell.V_m *= exp(-(event_time - t) / cell.tau_m);
        // Add jump due to spike.
        cell.V_m += weight/cell.C_m;

        t = event_time;

        // If crossing threshold occurred
        if (cell.V_m >= cell.V_th) {
            cell_member_type spike_neuron_gid = {gid_base_ + lid, 0};
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
