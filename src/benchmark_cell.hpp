#pragma once

#include <time_sequence.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::benchmark

struct benchmark_cell {
    // Describes the time points at which spikes are to be generated.
    time_seq time_sequence;

    // Time taken in μs to advance the cell one ms of simulation time.
    // If equal to 1, then a single cell can be advanced per 
    double run_time_per_ms;
};

} // namespace arb


