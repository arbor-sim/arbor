#pragma once
#pragma once

#include <common_types.hpp>
#include <util/debug.hpp>
#include <vector>

namespace nest {
namespace mc {

class fs_cell {
public:
    using index_type = cell_lid_type;
    using size_type = cell_local_size_type;
    using value_type = double;

    // constructor
    fs_cell(time_type start_time, time_type period, time_type stop_time):
        start_time_(start_time),
        period_(period),
        stop_time_(stop_time),
        time_(0.0)
    {

    }

    /// Return the kind of cell, used for grouping into cell_groups
    cell_kind const get_cell_kind() const {
        return cell_kind::regular_frequency;
    }


    std::vector<time_type> spikes_until(time_type tfinal)
    {
        std::vector<time_type> spike_times;

        // If we should be spiking in this 'period'
        if (tfinal > start_time_ &&
            (tfinal - period_) < stop_time_)
        {
            // Generate all possible spikes in this time frame (typically only one!)
            for (time_type time = start_time_ > time_ ? start_time_ : time_;
                time < tfinal;
                time += period_)
            {
                spike_times.push_back(time);
            }
        }

        // Save our current time we generate exclusive a possible tfinal spike
        time_ = tfinal;
        return spike_times;
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
