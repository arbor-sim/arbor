#pragma once

#include <common_types.hpp>

namespace arb {

/// Description class for a regular spike source: a cell that generates
/// spikes with a fixed period over a given time interval.

struct ips_cell {
    time_type start_time;
    time_type stop_time;

    // Rate of the Piosson process (in spikes per second)
    time_type rate;
    // Every sample_delta we sample if we should emit a spike (in ms)
    time_type sample_delta;
};

} // namespace arb
