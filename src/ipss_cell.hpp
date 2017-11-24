#pragma once

#include <utility>
#include <random>

#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell_description.hpp>
#include <util/unique_any.hpp>

namespace arb {

/// Cell implementation of inhomogeneous Poisson spike generator.
/// Spikes are generated at a configurable sample rate. With a rate that can be
/// varied across time. Rate can be piece wise constant or linearly interpolated
/// per sample time step.
class ipss_cell : public ipss_cell_description {
public:
    /// Constructor
    ipss_cell(ipss_cell_description desc, cell_gid_type gid) :
        ipss_cell_description(std::move(desc)), gid_(gid), time_(0.0),
        generator_(gid), prob_(0.0), prob_dt_(0.0) {
        // We now have ownership of the rate_vector add a single rate pair
        // At the end with the stop time_. We can now use an itterator to the
        // next rate change.
        rates_per_time.push_back({ stop_time + sample_delta, rates_per_time.back().second });

        reset();
    }

    cell_kind get_cell_kind() const {
        return cell_kind::inhomogeneous_poisson_spike_source;
    }

    /// reset internal variables based on the rate vector supplied add construction
    ///   - Find the first valid rate depending on the start time_
    ///   - Set the rates and rate change per step
    ///   - Set the iterator to the next rate change
    ///   - time zero
    ///   - reseed rng
    void reset() {
        time_ = 0.0;

        // Reset the random number generator!
        generator_.seed(gid_);

        // Sanity check: we need a rate at the start of the neuron time_
        if (rates_per_time.cbegin()->first > start_time) {
            throw std::logic_error("The start time of the neuron is before the first time/rate pair");
        }

        it_next_rate_ = rates_per_time.cbegin();
        prob_ = it_next_rate_->second;

        // loop over the entries until we have the last change before
        // the start time_ of the cell
        while (it_next_rate_->first <= start_time) {
            rate_change_step();
        }

    }

    /// Advance the Poisson generator until end of ep
    /// If a rate change is occured during this ep the new rate will be used
    /// including possible interpolation between steps.
    ///
    /// Floating point noise might result in a final step that ends a maximum of
    /// sample_delta - MIN(float) after ep. Internal timekeeping will take this
    /// in account in next steps.
    void advance(epoch ep, std::vector<spike>& spikes_)
    {
        // Get begin and start range: cell config and epoch ranges
        auto t = std::max(start_time, time_);
        auto t_end = std::min(stop_time, ep.tfinal);

        // if cell is not active skip
        if (t >= t_end) {
            return;
        }

        double prob = prob_;
        while (t < t_end) {
            // Do we run till the next rate change or till end of epoch?
            double t_end_step = it_next_rate_->first < t_end ?
                it_next_rate_->first : t_end;

            while (t < t_end_step) {
                // roll a dice between 0 and 1, if below prop we have a spike
                if (distribution_(generator_) < prob) {
                    spikes_.push_back({ { gid_, 0 }, t });
                }
                t += sample_delta;
                prob += prob_dt_;
            }

            // Did we have a rate change inside of the epoch?
            if (it_next_rate_->first < t_end) {
                // update the to the new rate
                rate_change_step();
            }
        }
        // Store for next epoch
        time_ = t;
        prob_ = prob;
    }
private:
    /// We need to do some book keeping when we have a rate_change step
    ///   - Set the new base rate based on the rate vector
    ///   - Update the next rate change it.
    ///   if we have interpolation:
    ///   - Calculate the rate_change per sample step size
    void rate_change_step() {
        // We need the start value for this rate change
        auto start_time = it_next_rate_->first;
        prob_ = (it_next_rate_->second / 1000.0) * sample_delta;

        it_next_rate_++;

        // If we interpolate we calculate the rate_change per sample_delta step
        // based on the next rate change iterator values
        if (interpolate) {
            double next_prob = (it_next_rate_->second / 1000.0) * sample_delta;
            unsigned steps = (it_next_rate_->first - start_time) / sample_delta;
            prob_dt_ = (next_prob - prob_) / steps;
        }
    }

    cell_gid_type gid_;

    // Internal time of the cell. Assures that we have continues time when
    // floating point noise occurs
    time_type time_;

    // Unique generator per cell
    std::mt19937 generator_;

    // The current rate  (spike/s) we are running at
    double prob_;

    // How much much of a rate of change per sampling time step needed for
    // interpolation.
    double prob_dt_;

    // iterator in the vector of time-rate pairs when to change to new rate
    std::vector<std::pair<time_type, double>>::const_iterator it_next_rate_;

    // Distribution for Poisson generation
    std::uniform_real_distribution<float> distribution_;
};
} // namespace arb

