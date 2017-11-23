#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

/// Cell group implementing RSS cells.



class ipss_cell_group: public cell_group {
public:
    ipss_cell_group(std::vector<cell_gid_type> gids, const recipe& rec) {
        cells_.reserve(gids.size());
        for (auto gid: gids) {
            cells_.emplace_back(
                util::any_cast<ipss_cell>(rec.get_cell_description(gid)), gid);
        }

        distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
        reset();
    }

    cell_kind get_cell_kind() const override {
        return cell_kind::inhomogeneous_poisson_spike_source;
    }

    void reset() override {
        clear_spikes();
        for (auto &cell: cells_) {
            cell.time = 0.0;

            // Reset the random number generator!
            cell.generator = std::mt19937(cell.gid);

            cell.initialize_cell_state();
        }
    }

    void set_binning_policy(binning_kind policy, time_type bin_interval) override {}

    void advance(epoch ep, time_type dt, const event_lane_subrange& events) override {
        for (auto& cell : cells_) {
            // Get begin and start range: cell config and epoch ranges
            auto t = std::max(cell.start_time, cell.time);
            auto t_end = std::min(cell.stop_time, ep.tfinal);

            // if cell is not active skip
            if (t >= t_end) {
                continue;
            }

            // The probability per sample step
            while (true)
            {
                // Do we run till end of epoch, or till the next rate change
                double t_end_step = cell.next_rate_change_it->first < t_end ?
                    cell.next_rate_change_it->first : t_end;

                double prob_per_dt = (cell.current_rate / 1000.0) * cell.sample_delta;

                // Float noise might result in a final step larger then t_end.
                while (t < t_end_step)
                {
                    // roll a dice between 0 and 1, if below prop we have a spike
                    if (distribution(cell.generator) < prob_per_dt) {
                        spikes_.push_back({ { cell.gid, 0 }, t });
                    }
                    t += cell.sample_delta;
                }

                // Did we have a rate change inside of the epoch?
                if (cell.next_rate_change_it->first < t_end) {
                    // update the to the new rate
                    cell.current_rate = cell.next_rate_change_it->second;

                    // increase the next_rate_change_it pointer
                    cell.next_rate_change_it++;
                }
                else {
                    break;
                }
            }
            cell.time = t;
        }
    }

    const std::vector<spike>& spikes() const override {
        return spikes_;
    }

    void clear_spikes() override {
        spikes_.clear();
    }

    void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) override {
        std::logic_error("pi_cell does not support sampling");
    }

    void remove_sampler(sampler_association_handle) override {}

    void remove_all_samplers() override {}

private:
    // RSS description plus gid for each RSS cell.
    struct ipss_info: public ipss_cell {
        ipss_info(ipss_cell desc, cell_gid_type gid):
            ipss_cell(std::move(desc)),
            gid(gid),
            time(0.0),
            generator(gid),
            current_rate(0.0) {
            // We now have ownership of the rate_vector add a single rate pair
            // At the end with the stop time. We can now use an itterator to the
            // next rate change.
            rates_per_time.push_back({ stop_time, rates_per_time.back().second });

            initialize_cell_state();
        }

        void initialize_cell_state() {
            // Sanity check: we need a rate at the start of the neuron time
            // Todo: move to constructor?
            if (rates_per_time.cbegin()->first > start_time) {
                throw std::logic_error("The start time of the neuron is before the first time/rate pair");
            }

            next_rate_change_it = rates_per_time.cbegin();
            current_rate = next_rate_change_it->second;
            // loop over the entries until we have the last change before
            // the start time of the cell
            while (next_rate_change_it->first <= start_time) {
                current_rate = next_rate_change_it->second;
                next_rate_change_it++;
            }
        }

        cell_gid_type gid;
        time_type time;

        // Each cell has a unique generator
        std::mt19937 generator;

        // The current rate  (spike/s) we are running at
        double current_rate;

        // pointer into a vector of time rate pairs when to change to new rates
        std::vector<std::pair<time_type, double>>::const_iterator next_rate_change_it;
    };

    // RSS cell descriptions.
    std::vector<ipss_info> cells_;

    // Spikes that are generated.
    std::vector<spike> spikes_;

    // Distribution for Poisson generation
    std::uniform_real_distribution<float> distribution;
};
} // namespace arb

