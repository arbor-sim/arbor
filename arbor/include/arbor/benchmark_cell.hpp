#pragma once

#include <arbor/export.hpp>
#include <arbor/schedule.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::benchmark

struct ARB_SYMBOL_VISIBLE benchmark_cell {
    cell_tag_type source; // Label of source.
    cell_tag_type target; // Label of target.

    // Describes the time points at which spikes are to be generated.
    schedule time_sequence;

    // Time taken in ms to advance the cell one ms of simulation time.
    // If equal to 1, then a single cell can be advanced in realtime 
    double realtime_ratio;

    benchmark_cell() = delete;
    benchmark_cell(cell_tag_type source, cell_tag_type target, schedule seq, double ratio):
        source(source), target(target), time_sequence(seq), realtime_ratio(ratio) {};
};

} // namespace arb


