#include <lif_cell_group_mc.hpp>

using namespace arb;

// Constructor containing gid of first cell in a group and a container of all cells.
lif_cell_group_mc::lif_cell_group_mc(std::vector<cell_gid_type> gids, const recipe& rec):
gids_(std::move(gids))
{
    // Default to no binning of events
    set_binning_policy(binning_kind::none, 0);

    // Build lookup table for gid to local index.
    for (auto i: util::make_span(0, gids_.size())) {
      gid_index_map_[gids_[i]] = i;
    }

    // reserve
    cells_.reserve(gids_.size());

    // resize
    next_queue_event_index.resize(gids_.size());
    lambda_.resize(gids_.size());
    next_poiss_time_.resize(gids_.size());
    cell_events_.resize(gids_.size());
    last_time_updated_.resize(gids_.size());
    poiss_event_counter_ = std::vector<unsigned>(gids_.size());

    // Initialize variables for the external Poisson input.
    for (auto lid: util::make_span(0, gids_.size())) {
        cells_.push_back(util::any_cast<lif_cell_description>(rec.get_cell_description(gids_[lid])));

        // If a cell receives some external Poisson input then initialize the corresponding variables.
        if (cells_[lid].n_poiss > 0) {
            EXPECTS(cells_[lid].n_poiss >= 0);
            EXPECTS(cells_[lid].w_poiss >= 0);
            EXPECTS(cells_[lid].d_poiss >= 0);
            EXPECTS(cells_[lid].rate >= 0);
            auto rate = cells_[lid].rate * cells_[lid].n_poiss;
            lambda_[lid] = 1.0 / rate;
            sample_next_poisson(lid);
        }
    }
}

cell_kind lif_cell_group_mc::get_cell_kind() const {
    return cell_kind::lif_neuron;
}

void lif_cell_group_mc::advance(epoch ep, time_type dt, const event_lane_subrange& event_lanes) {
    PE("lif");
    if (event_lanes.size()) {
      for (auto lid: util::make_span(0, gids_.size())) {
        // Advance each cell independently.
        next_queue_event_index[lid] = 0;
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
    // Clear all the event queues.
    for (auto& queue : cell_events_) {
        queue.clear();
    }
    next_poiss_time_.clear();
    poiss_event_counter_.clear();
    last_time_updated_.clear();
    next_queue_event_index.clear();
}

// Samples next poisson spike.
void lif_cell_group_mc::sample_next_poisson(cell_gid_type lid) {
    // key = GID of the cell
    // counter = total number of Poisson events seen so far
    auto key = gids_[lid] + 1225;
    auto counter = poiss_event_counter_[lid];
    ++poiss_event_counter_[lid];

    // First sample unif~Uniform(0, 1) and then use it to get the Poisson distribution
    time_type t_update = random_generator::sample_poisson(lambda_[lid], counter, key);

    next_poiss_time_[lid] += t_update;
}

// Returns the time of the next poisson event for given neuron,
// taking into accout the delay of poisson spikes,
// without sampling a new Poisson event time.
template <typename Pred>
util::optional<time_type> lif_cell_group_mc::next_poisson_event(cell_gid_type lid, time_type tfinal, Pred should_pop) {
    if (cells_[lid].n_poiss > 0) {
        time_type t_poiss =  next_poiss_time_[lid] + cells_[lid].d_poiss;
        return should_pop(t_poiss,tfinal) ? util::optional<time_type>(t_poiss) : util::nullopt;
    }
    return util::nullopt;
}

template <typename Pred>
util::optional<postsynaptic_spike_event> pop_if(pse_vector& event_lane, unsigned& start_index, time_type tfinal, Pred should_pop) {
    if (event_lane.size() <= start_index || !should_pop(event_lane[start_index].time, tfinal)) {
        return util::nullopt;
    }
    // instead of deleting this event from the queue, just increase the starting index
    auto ev = event_lane[start_index];
    start_index++;
    return ev;
}

// Returns the next most recent event that is yet to be processed.
// It can be either Poisson event or the queue event.
// Only events that happened before tfinal are considered.
template <typename Pred>
util::optional<postsynaptic_spike_event> lif_cell_group_mc::next_event(cell_gid_type lid, time_type tfinal, pse_vector& event_lane, Pred should_pop) {
    if (auto t_poiss = next_poisson_event(lid, tfinal, should_pop)) {
        // if (auto ev = cell_events_[lid].pop_if(std::min(tfinal, t_poiss.get()))) {
        if (auto ev = pop_if(event_lane, next_queue_event_index[lid], tfinal, should_pop)) {
            return ev;
        }
        sample_next_poisson(lid);
        return postsynaptic_spike_event{{cell_gid_type(gids_[lid]), 0}, t_poiss.value(), cells_[lid].w_poiss};
    }

    // t_queue < tfinal < t_poiss => return t_queue
    // return cell_events_[lid].pop_if(tfinal);
    return pop_if(event_lane, next_queue_event_index[lid], tfinal, should_pop);
}

// Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
// Parameter dt is ignored, since we make jumps between two consecutive spikes.
void lif_cell_group_mc::advance_cell(time_type tfinal, time_type dt, cell_gid_type lid, pse_vector& event_lane) {
    // Current time of last update.
    auto t = last_time_updated_[lid];
    auto& cell = cells_[lid];

    // If a neuron was in the refractory period,
    // ignore any new events that happened before t,
    // including poisson events as well.
    while (auto ev = next_event(lid, t, event_lane, [](time_type lhs, time_type rhs) -> bool {return lhs < rhs;})) {
    };

    // Integrate until tfinal using the exact solution of membrane voltage differential equation.
    while (auto ev = next_event(lid, tfinal, event_lane, [](time_type lhs, time_type rhs) -> bool {return lhs < rhs;})) {
        auto weight = ev->weight;
        auto event_time = ev->time;

        // If a neuron is in refractory period, ignore this event.
        if (event_time < t) {
            continue;
        }

        // if there are events that happened at the same time as this event, process them as well
        while (auto coinciding_event = next_event(lid, event_time, event_lane, [](time_type lhs, time_type rhs) -> bool {return lhs <= rhs;})) {
            weight += coinciding_event->weight;
        }

        // Let the membrane potential decay.
        auto decay = exp(-(event_time - t) / cell.tau_m);
        cell.V_m *= decay;
        auto update = weight / cell.C_m;
        // Add jump due to spike.
        cell.V_m += update;
        t = event_time;

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
