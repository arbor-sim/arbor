#pragma once
#pragma once

#include <vector>

#include <common_types.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {

class fs_cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;

    /// Create a frequency spiker: A cell that generated spikes with a certain
    /// period for a set time range.
    fs_cell(time_type start_time=0.0, time_type period=1.0, time_type stop_time=0.0):
        start_time_(start_time),
        period_(period),
        stop_time_(stop_time),
        time_(0.0)
    {
    }

    /// Return the kind of cell, used for grouping into cell_groups
    cell_kind  get_cell_kind() const  {
        return cell_kind::regular_spike_source;
    }

    /// Collect all spikes until tfinal.
    // updates the internal time state to tfinal as a side effect
    std::vector<time_type> spikes_until(time_type tfinal)
    {
        std::vector<time_type> spike_times;

        // If we should be spiking in this 'period', this is the main
        // optimization of this function.
        if (tfinal > start_time_) {
            // We have to spike till tfinal or the stop_time_of the neuron
            auto end_time = stop_time_ < tfinal ? stop_time_ : tfinal;
            // Generate all possible spikes in this time frame typically only
            for (time_type time = start_time_ > time_ ? start_time_ : time_;
                time < end_time;
                time += period_) {
                spike_times.push_back(time);
            }

        }
        // Save our current time we generate exclusive a possible tfinal spike
        time_ = tfinal;


        return spike_times;
    }

    /// reset internal time to 0.0
    void reset()  {
        time_ = 0.0;
    }

private:
    // When to start spiking
    time_type start_time_;

    // with what period
    time_type period_;

    // untill when
    time_type stop_time_;

    // internal time, storage
    time_type time_;
};

} // namespace mc
} // namespace nest
