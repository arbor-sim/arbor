#pragma once

#include <utility>
#include <random>
#include <limits>

#include <cell_group.hpp>
#include <recipe.hpp>
#include <ipss_cell_description.hpp>
#include <util/unique_any.hpp>


namespace arb {

/// Cell implementation of inhomogeneous Poisson spike generator.
/// Spikes are generated at a configurable sample rate. With a rate that can be
/// varied across time. Rate can be piece wise constant or linearly interpolated
/// per sample time step.
/// All times supplied to the cell will be 'rounded' to the first higher multiple of
/// of the supplied sample_delta.
/// When interpolating the rate will be the averaged rate between the
/// t and t+1. The return spike time will be t.
/// The first rate supplied in the time-rate vector should be at or before the
/// start time of the cell. The rate of the last supplied time-rate will be kept
/// until the stop_time of the cell

class ipss_cell   {
public:
    /// Constructor
    ipss_cell(ipss_cell_description desc, cell_gid_type gid) :
        gid_(gid), sample_delta_(desc.sample_delta),
        start_step_(time_to_step_idx(desc.start_time, sample_delta_)),
        stop_step_(time_to_step_idx(desc.stop_time, sample_delta_)),
        current_step_(0), interpolate_(desc.interpolate),
        generator_(gid_)
    {
        // For internal storage convert the rates from spikes / second to
        // spikes per sample_delta_  and the time to step_idx from t = 0.0
        for (auto entry : desc.rates_per_time) {
            auto spikes_per_sample_delta = (entry.second / 1000.0) * sample_delta_;
            auto time_step = time_to_step_idx(entry.first, sample_delta_);
            rates_per_step_.push_back({ time_step, spikes_per_sample_delta });
        }

        // Add and extra rate entry at max long, copy of the last original entry
        // This allows current and next iterator.
        rates_per_step_.push_back({ -1, rates_per_step_.cend()->second });

        // Sanity checks
        if (start_step_ > stop_step_) {
            throw std::logic_error("The start time is after the end time in the Inhomogeneous Poisson Spike Source.");
        }

        if (start_step_ < rates_per_step_.cbegin()->first) {
            throw std::logic_error("The start time of the Inhomogeneous Poisson Spike Source is before the first time/rate pair");
        }

        reset();
    }

    cell_kind get_cell_kind() const {
        return cell_kind::inhomogeneous_poisson_spike_source;
    }

    /// reset cell to start state
    ///   - Find the first valid rate depending on the start time_
    ///   - Set the rates based on the rate change iterators
    ///   - Reseed the rng
    void reset() {

        // set the pointer to the first and second entry in the rates vector
        current_rate_it_ = rates_per_step_.cbegin();
        next_rates_it_ = current_rate_it_;
        next_rates_it_++;
        update_rates();

        // Step through the rate_change vector if begin_step does not fall in the
        // range of the current pointers
        while (start_step_ >= next_rates_it_->first) {
            // updates the current rates and dt
            current_rate_it_++;
            next_rates_it_++;
            update_rates();
        }

        // Set the internal step counter to the start of the cell
        current_step_ = start_step_;

        // Reset the random number generator!
        generator_.seed(gid_);
    }

    /// Advance the Poisson generator until end of epoch
    /// If a rate change is occurred during this ep the new rate will be used
    /// including possible interpolation between steps.
    void advance(epoch ep, std::vector<spike>& spikes)
    {
        // Convert the epoch end to step
        unsigned long epoch_end_step = time_to_step_idx(ep.tfinal, sample_delta_);

        // If the start time of the neuron is after the end of the epoch
        if (start_step_ > epoch_end_step) {
            return;
        }

        auto end_step = std::min(stop_step_, epoch_end_step);

        // epoch is after end of cell
        if (current_step_ >= end_step) {
            return;
        }

        // We are in an active epoch, start stepping!
        do {
            // Should we change the rates in the current step?

            if (current_step_ == next_rates_it_->first) {
                // update the itterators
                current_rate_it_++;
                next_rates_it_++;
                update_rates();
            }

            // We are working with small probabilities use double
            // rate = base rate + nr steps since start rate * delta per step
            double spike_rate = base_rate_ + (current_step_ - current_rate_it_->first) * rate_delta_step_ ;

            // This dice roll takes 93% of the runtime of this cell.
            auto dice_roll = distribution_(generator_);
            if (dice_roll < spike_rate) {
                spikes.push_back({ { gid_, 0 }, current_step_ * sample_delta_ });
            }

        } while (++current_step_ < end_step);

    }

    // Small helper function for 'rounding' up to whole multiples of dt
    // public for testing
    static  unsigned long time_to_step_idx(time_type time, time_type dt) {
        unsigned long whole_multiples = static_cast<unsigned long>(time / dt);

        if (whole_multiples * dt < time) {
            return whole_multiples + 1;
        }

        return whole_multiples;
    }

private:
    /// We need to do some book keeping when we have a rate_change step
    ///   - Calculate the base rate and the delta per step
    void update_rates() {
        // First calculate the rate per time step (if we interpolate)
        auto steps_this_rate = next_rates_it_->first - current_rate_it_->first;
        auto rate_delta_this_range = next_rates_it_->second - current_rate_it_->second;

        rate_delta_step_ = interpolate_ ? rate_delta_this_range / steps_this_rate : 0.0;

        // The base rate is the rate at the 0th step in this range plus
        // half a rate_delta_step_
        base_rate_ = current_rate_it_->second + 0.5 * rate_delta_step_;
    }

    cell_gid_type gid_;

    // Size of time steps we take in the cell
    time_type sample_delta_;

    // Start and stop step of the cell
    unsigned long start_step_;
    unsigned long stop_step_;

    // Which step we are currently are at
    unsigned long current_step_;

    // Every sample_delta_ we sample if we should emit a spike (in ms)
    bool interpolate_;

    using time_rate_type = std::pair<unsigned long, double>;

    // Vector of time_step rate pairs
    std::vector<time_rate_type> rates_per_step_;

    // itterators to the current rate and the next rate
    std::vector<time_rate_type>::const_iterator current_rate_it_;
    std::vector<time_rate_type>::const_iterator next_rates_it_;


    // Unique generator per cell
    std::mt19937 generator_;

    // The current rates (spike / sample step)
    double base_rate_;
    double rate_delta_step_;

    // Distribution for Poisson generation
    std::uniform_real_distribution<float> distribution_;
};
} // namespace arb

