#pragma once

#include <vector>

namespace arb {

// Delay-aware spike exchange via MPI neighbourhood collectives.
//
// By default, all spikes are exchanged with a global all-to-all after every
// epoch, where an epoch is half the minimum delay of the whole network. With
// this option, the ranks that the calling rank receives spikes from are split
// into levels instead:
//
//   * the calling rank itself, which is handled without communication,
//   * one level per entry in `neighborhoods`, exchanged with a sparse MPI
//     neighbourhood collective over just those ranks,
//   * all remaining ("far") ranks, exchanged with a global all-to-all that sends
//     each spike only to the ranks with connections from its source.
//
// Spikes are buffered per level and only sent as often as the minimum delay of
// the connections from that level permits. If, for example, connections between
// neighbouring ranks are much shorter than connections between distant ones,
// the expensive global collective runs only every few epochs. All buffers are
// flushed at the end of each call to simulation::run, so the global spike count
// is exact after run returns.
struct neighborhood_spike_exchange {
    // Ranks this rank receives spikes from, grouped into levels, innermost
    // first. A rank listed in more than one level belongs to the first; the
    // calling rank is ignored. Nested levels, where each level repeats the
    // previous one, are fine. Every rank must pass the same number of levels.
    // Without any levels, only the delay-aware buffering of far spikes is used.
    std::vector<std::vector<int>> neighborhoods;
};

} // namespace arb
