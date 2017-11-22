#pragma once

#include <common_types.hpp>

namespace arb {

/// Description class for a regular spike source: a cell that generates
/// spikes with a fixed period over a given time interval.

struct pi_cell {
    time_type start_time;
    time_type stop_time;

    // Rate of the Piosson process (in spikes per second)
    time_type rate;
};

} // namespace arb
