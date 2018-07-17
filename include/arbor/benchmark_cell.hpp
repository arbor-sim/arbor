#pragma once

#include <arbor/schedule.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::benchmark

struct benchmark_cell {
    // Describes the time points at which spikes are to be generated.
    schedule time_sequence;

    // Time taken in ms to advance the cell one ms of simulation time.
    // If equal to 1, then a single cell can be advanced in realtime 
    double realtime_ratio;
};

} // namespace arb


