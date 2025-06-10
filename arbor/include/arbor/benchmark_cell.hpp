#pragma once

#include <arbor/export.hpp>
#include <arbor/schedule.hpp>

template<typename K>
void serialize(arb::serializer& s, const K& k, const arb::schedule&);
template<typename K>
void deserialize(arb::serializer& s, const K& k, arb::schedule&);

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
    double realtime_ratio = 1.0;

    ARB_SERDES_ENABLE(benchmark_cell, source, target, time_sequence, realtime_ratio);
};

using benchmark_cell_editor = std::function<void(benchmark_cell&)>;

} // namespace arb
