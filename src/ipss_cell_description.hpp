#pragma once

#include <common_types.hpp>

namespace arb {

/// Description class for a regular spike source: a cell that generates
/// spikes with a fixed period over a given time interval.

struct ipss_cell_description {
    time_type start_time;
    time_type stop_time;

    // Every sample_delta we sample if we should emit a spike (in ms)
    double sample_delta;

    // vector of spike_rates starting at times
    // The vector needs at least a single entry
    std::vector<std::pair<time_type, double>> rates_per_time;
    bool interpolate;

    // rates_per_time: A vector of spike_rates each starting at the supplied time
    // The first time-rate pair should have a time before the start_time
    ipss_cell_description(time_type start_time, time_type stop_time, time_type sample_delta,
        std::vector<std::pair<time_type, double>> rates_per_time, bool interpolate = true) :
        start_time(start_time),
        stop_time(stop_time),
        sample_delta(sample_delta),
        rates_per_time(std::move(rates_per_time)),
        interpolate(interpolate)
    {}

};

} // namespace arb
