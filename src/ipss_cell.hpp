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


class ipss_cell   {
public:
    /// Constructor
    ipss_cell(ipss_cell_description desc, cell_gid_type gid) :
        gid_(gid), sample_delta_(desc.sample_delta),
        start_step_(time_to_step_idx(desc.start_time, sample_delta_)),
        stop_step_(time_to_step_idx(desc.stop_time, sample_delta_)),
        current_step_(0), interpolate_(desc.interpolate),
        generator_(gid)
    {
        // For internal storage convert the rates from spikes / second to
        // spikes per sample_delta_ and the times to sample_delta_ time step since
        // t = 0.0
        for (auto entry : desc.rates_per_time) {
            auto spikes_per_sample_delta = (entry.second / 1000.0) * sample_delta_;
            auto time_step = time_to_step_idx(entry.first, sample_delta_);
            rates_per_step_.push_back({ time_step, spikes_per_sample_delta });
        }

        // Add and extra rate entry at max long, copy of the last orignal entry
        rates_per_step_.push_back({ -1, rates_per_step_.cend()->second });

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
    ///   - Interpolate start rate
    void reset() {
        // Sanity check: we need a rate at the start of the neuron time_
        if (start_step_ < rates_per_step_.cbegin()->first) {
            throw std::logic_error("The start time of the neuron is before the first time/rate pair");
        }

        // Internal time of the cell to the zero
        current_step_ = 0;

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

        // Reset the random number generator!
        generator_.seed(gid_);
    }

    /// Advance the Poisson generator until end of epoch
    /// If a rate change is occurred during this ep the new rate will be used
    /// including possible interpolation between steps.
    ///
    void advance(epoch ep, std::vector<spike>& spikes)
    {
        // Convert the epoch end to step
        unsigned long epoch_end_step = time_to_step_idx(ep.tfinal, sample_delta_);
        std::cout << "epoch_end_step: " << epoch_end_step  << "\n";
        // If we have started
        if (epoch_end_step > start_step_) {
            current_step_ = epoch_end_step;
            return;
        }

        auto end_step = std::min(stop_step_, epoch_end_step);

        std::cout << "end_step: " << end_step << "\n";

        // epoch is after end of cell
        if (current_step_ >= end_step) {
            current_step_ = end_step;
            return;
        }

        std::cout << "base: "<< base_rate_ << "," << rate_delta_step_ << "\n";
        // We are in an active epoch, start stepping!
        do {
            // Should we change the rates in the current step?
            if (current_step_ == next_rates_it_->first) {
                // update the itterators
                current_rate_it_++;
                next_rates_it_++;
                update_rates();
            }
            double spike_rate = base_rate_ + (current_step_ - current_rate_it_->first) * rate_delta_step_ ;
            auto dice_roll = distribution_(generator_);
            std::cout << spike_rate << "," << spike_rate << "\n";
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
    ///   - Set the new base rate based on the rate vector
    ///   - Update the next rate change it.
    ///   if we have interpolation:
    ///   - Calculate the rate_change per sample step size
    void update_rates() {


        // First calculate the rate per time step (if we interpolate)
        auto steps_this_rate = next_rates_it_->first - current_rate_it_->first;
        auto rate_delta_this_range = (next_rates_it_->second - current_rate_it_->second);

        std::cout << "steps_this_rate: " << steps_this_rate << "," << rate_delta_this_range << "\n";
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

