#pragma once

#include <cell_group.hpp>
#include <event_queue.hpp>
#include <lif_cell_description.hpp>
#include <profiling/profiler.hpp>
#include <random>
#include <util/unique_any.hpp>
#include <vector>

namespace nest {
namespace mc {

class lif_cell_group: public cell_group {
public:

    using value_type = double;

    lif_cell_group() = default;

    // Constructor containing gid of first cell in a group and a container of all cells.
    lif_cell_group(cell_gid_type first_gid, const std::vector<util::unique_any>& cells):
    gid_base_(first_gid)
    {
        cells_.reserve(cells.size());
        lambda_.reserve(cells.size());

        generator_.resize(cells.size());
        next_poiss_time_.resize(cells.size());

        cell_events_.resize(cells.size());
        last_time_updated_.resize(cells.size());

        // Cast each cell to lif_cell_description.
        for (const auto& cell : cells) {
            cells_.push_back(util::any_cast<lif_cell_description>(cell));
        }

        //std::cout << "cells_.size() = " << cells_.size() << "\n";
        
        // Initialize variables for the external Poisson input.
        for (auto lid : util::make_span(0, cells_.size())) {
            EXPECTS(cells_[lid].n_poiss >= 0);
            EXPECTS(cells_[lid].w_poiss >= 0);
            EXPECTS(cells_[lid].d_poiss >= 0);
            EXPECTS(cells_[lid].rate >= 0);
            
            if (cells_[lid].n_poiss > 0) {
                lambda_.push_back(1.0/(cells_[lid].rate * cells_[lid].n_poiss));
                generator_[lid].seed(1000 + first_gid + lid);
                next_poiss_time_[lid] = exp_dist_(generator_[lid]) * lambda_[lid];
            }
        }

    }

    cell_kind get_cell_kind() const override {
        return cell_kind::lif_neuron;
    }

    // TODO: Remove this after testing.
    // Samples voltage from time t to tfinal with resolution
    // given by class parameter sampling_dt_
    void sample_voltage(value_type v, value_type E_L, value_type tau_m, time_type tend, bool refractory_period) {
        time_type start = last_time_voltage_updated_;

        for (time_type time = last_time_voltage_updated_; time < tend; time += sampling_dt_) {
            if (!refractory_period) {
                v *= exp(-(time - last_time_voltage_updated_)/tau_m);
            }
            else {
                // If in refractory period, then at the moment of spike,
                // let v has original value (which is >= V_thr) and
                // otherwise, let v be equal to the resting potential E_L.
                if (time > start) {
                    v = E_L;
                }
            }
            // Put a pair of time and voltage.
            voltage_.push_back({time, v});
            // Update last time the voltage is updated.  last_time_voltage_updated_ = time;
        }
    }

    void advance(time_type tfinal, time_type dt) override {
        PE("lif");
        // Distribute incoming events to individual cells.
        while (!events_.empty()) {
            // Take event from the queue and pop it.
            auto ev = events_.front();
            events_.pop();

            int target_gid = ev.target.gid;
            // Transform gid -> lid.
            int lid = target_gid - gid_base_;

            if (cells_[lid].n_poiss > 0) {
                // Generate all the Poisson events that happened before this event.
                while (next_poiss_time_[lid] < ev.time) {
                    //std::cout << "generating poisson spikes until ev.time" << std::endl;
                    // Generate a Poisson event.
                    postsynaptic_spike_event poiss_event = {{cell_gid_type(target_gid), 0}, next_poiss_time_[lid] + cells_[lid].d_poiss, cells_[lid].w_poiss};
                    cell_events_[lid].push(poiss_event);
                    // Sample next Poisson event.
                    next_poiss_time_[lid] += exp_dist_(generator_[lid]) * lambda_[lid];
                }
            }
            cell_events_[lid].push(ev);
        }

        for (size_t lid = 0; lid < cells_.size(); ++lid) {
            if (cells_[lid].n_poiss > 0) {
                // Generate Poisson events until tfinal is reached.
                while (next_poiss_time_[lid] < tfinal) {
                    // Generate a Poisson event.
                    postsynaptic_spike_event poiss_event = {{cell_gid_type(lid) + gid_base_, 0}, next_poiss_time_[lid] + cells_[lid].d_poiss, cells_[lid].w_poiss};
                    cell_events_[lid].push(poiss_event);
                    //std::cout << "generating poisson spikes until tfinal " << poiss_event << std::endl;
                    // Sample next Poisson event.
                    next_poiss_time_[lid] += exp_dist_(generator_[lid]) * lambda_[lid];
                }
            }
            // Advance each cell independently.
            advance_cell(tfinal, dt, lid);
        }
        PL();
    }

    void enqueue_events(const std::vector<postsynaptic_spike_event>& events) override {
        for (auto& e: events) {
            events_.push(e);
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    // TODO: implement sampler
    void add_sampler(cell_member_type probe_id, sampler_function s, time_type start_time = 0) override {}

    // TODO: implement binner_
    void set_binning_policy(binning_kind policy, time_type bin_interval) override {
    }

    // no probes in single-compartment cells
    std::vector<probe_record> probes() const override {
        return {};
    }

    void reset() override {
        spikes_.clear();
        // STL queue does not support clear()
        events_ = decltype(events_)();

        // TODO: Remove after testing.
        voltage_.clear();
    }

    // TODO: Remove after testing.
    std::vector<std::pair<time_type, value_type> > voltage() {
        return voltage_;
    }

    // TODO: Remove after testing.
    // Used to set up the resolution of sampling voltage.
    void turn_on_sampling(time_type resolution) {
        sampling_dt_ = resolution;
    }

private:
    // Gid of first cell in group.
    cell_gid_type gid_base_;

    // Cells that belong to this group.
    std::vector<lif_cell_description> cells_;

    // Spikes that are generated (not necessarily sorted).
    std::vector<spike> spikes_;

    // Pending events to be delivered.
    std::queue<postsynaptic_spike_event> events_;

    // Pending events per cell.
    std::vector<event_queue<postsynaptic_spike_event> > cell_events_;

    // Time when the cell was last updated.
    std::vector<time_type> last_time_updated_;

    //TODO: Remove after testing.
    // Sampling resolution.
    time_type sampling_dt_ = 0;

    // Assuming we have only 1 neuron in the group that
    // we want to track the voltage of.
    std::vector<std::pair<time_type, value_type> > voltage_;

    // Last time voltage was updated.
    time_type last_time_voltage_updated_ = 0;

    // External spike generation.

    // lambda_ = 1/(n_poiss * rate).
    std::vector<double> lambda_;

    // Random number generators.
    // Each cell has a separate generator in order to achieve the independence.
    std::vector<std::mt19937> generator_;

    // Unit exponential distribution (with mean 1).
    std::exponential_distribution<time_type> exp_dist_ = std::exponential_distribution<time_type>(1.0);

    // Sampled next Poisson event time for each cell.
    std::vector<time_type> next_poiss_time_;

    // Advances a single cell (lid) with the exact solution (jumps can be arbitrary).
    // Parameter dt is ignored, since we make a jumps between two consecutive spikes.
    void advance_cell(time_type tfinal, time_type dt, cell_gid_type lid) {
        // Current time of last update.
        auto t = last_time_updated_[lid];
        auto& cell = cells_[lid];

        // If a neuron was in the refractory period,
        // ignore any new events that happened before t.
        while (cell_events_[lid].pop_if_before(t));

        // Integrate until tfinal using the exact solution of membrane voltage differential equation.
        while (auto ev = cell_events_[lid].pop_if_before(tfinal)) {
            //std::cout << "received event : " << ev.get() << std::endl;
            auto weight = ev->weight;
            auto spike_time = ev->time;

            // If a neuron is in refractory period, ignore this spike.
            if (spike_time < t) {
                continue;
            }

            // TODO: Remove this after testing
            // used just for sampling the voltage.
            if (sampling_dt_ > 0 && gid_base_ == 0 && lid == 0) {
                sample_voltage(cell.V_m, cell.E_L, cell.tau_m, spike_time, false);
            }

            // Let the membrane potential decay.
            cell.V_m *= exp(-(spike_time - t) / cell.tau_m);
            // Add jump due to spike.
            cell.V_m += weight/cell.C_m;

            t = spike_time;

            // If crossing threshold occurred
            if (cell.V_m >= cell.V_th) {
                //std::cout << "spike occurred!" << std::endl;
                cell_member_type spike_neuron_gid = {gid_base_ + lid, 0};
                spike s = {spike_neuron_gid, spike_time};

                spikes_.push_back(s);

                // TODO: Remove this after testing!
                // Used just for sampling voltage.
                if (sampling_dt_ > 0 && gid_base_ == 0 && lid == 0) {
                    sample_voltage(cell.V_m, cell.E_L, cell.tau_m, t + cell.t_ref, true);
                }

                // Advance last_time_updated.
                t += cell.t_ref;

                // Reset the voltage to resting potential.
                cell.V_m = cell.E_L;
            }
            // This is the last time a cell was updated.
            last_time_updated_[lid] = t;
        }
    }

};
} // namespace mc
} // namespace nest
