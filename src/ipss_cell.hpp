#pragma once

#include <utility>
#include <random>


#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell_description.hpp>
#include <util/unique_any.hpp>


#include <iostream>

namespace arb {

    /// Cell group implementing RSS cells.
class ipss_cell : public ipss_cell_description {
public:

    ipss_cell(ipss_cell_description desc, cell_gid_type gid) :
        ipss_cell_description(std::move(desc)),
        gid(gid),
        time(0.0),
        generator(gid),
        current_prob(0.0),
        current_prob_dt(0.0)
    {
        // We now have ownership of the rate_vector add a single rate pair
        // At the end with the stop time. We can now use an itterator to the
        // next rate change.
        rates_per_time.push_back({ stop_time + sample_delta, rates_per_time.back().second });

        initialize_cell_state();
    }


    void initialize_cell_state() {
        // Sanity check: we need a rate at the start of the neuron time
        if (rates_per_time.cbegin()->first > start_time) {
            throw std::logic_error("The start time of the neuron is before the first time/rate pair");
        }

        next_rate_change_it = rates_per_time.cbegin();
        current_prob = next_rate_change_it->second;

        // loop over the entries until we have the last change before
        // the start time of the cell
        while (next_rate_change_it->first <= start_time) {
            double last_time = next_rate_change_it->first;

            rate_change_step(last_time);
        }
    }
    void rate_change_step(time_type last_time) {
        current_prob =
            (next_rate_change_it->second / 1000.0) * sample_delta;

        // increase the next_rate_change_it pointer

        next_rate_change_it++;
        if (interpolate)
        {
            double next_prob = (next_rate_change_it->second / 1000.0) * sample_delta;
            unsigned steps = (next_rate_change_it->first - last_time) / sample_delta;

            current_prob_dt = (next_prob - current_prob) / steps;
        }
    }

    void reset() {
        time = 0.0;

        // Reset the random number generator!
        generator = std::mt19937(gid);

        initialize_cell_state();

    }

    void advance(epoch ep, std::vector<spike>& spikes_)
    {
        // Get begin and start range: cell config and epoch ranges
        auto t = std::max(start_time, time);
        auto t_end = std::min(stop_time, ep.tfinal);

        // if cell is not active skip
        if (t >= t_end) {
            return;
        }

        double prob = current_prob;
        while (t < t_end) {
            // Do we run till end of epoch, or till the next rate change
            double t_end_step = next_rate_change_it->first < t_end ?
                next_rate_change_it->first : t_end;

            // Float noise might result in a final step larger then t_end.
            while (t < t_end_step)
            {
                // roll a dice between 0 and 1, if below prop we have a spike
                if (distribution(generator) < prob) {
                    spikes_.push_back({ { gid, 0 }, t });
                }
                t += sample_delta;
                prob += current_prob_dt;
            }

            // Did we have a rate change inside of the epoch?
            if (next_rate_change_it->first < t_end) {
                // update the to the new rate
                double last_time = next_rate_change_it->first;
                rate_change_step(last_time);
            }
        }
        time = t;
        current_prob = prob;
    }


private:
    cell_gid_type gid;
    time_type time;

    // Each cell has a unique generator
    std::mt19937 generator;

    // The current rate  (spike/s) we are running at
    double current_prob;

    double current_prob_dt;

    // pointer into a vector of time rate pairs when to change to new rates
    std::vector<std::pair<time_type, double>>::const_iterator next_rate_change_it;

    // Distribution for Poisson generation
    std::uniform_real_distribution<float> distribution;
};
} // namespace arb

