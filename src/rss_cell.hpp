#pragma once

#include <common_types.hpp>

namespace nest {
namespace mc {

/// Description class for a regular spike source: a cell that generates
/// spikes with a fixed period over a given time interval.

struct rss_cell {
    time_type start_time;
    time_type period;
    time_type stop_time;
};

} // namespace mc
} // namespace nest
