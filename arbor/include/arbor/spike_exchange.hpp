#pragma once

#include <cstddef>
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
//     neighbourhood all-gather over just those ranks,
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

    // Choose the levels from the connection delays instead; neighborhoods must
    // then be empty. Every rank must set this alike.
    //
    // Every source rank goes into a class by the exchange period its minimum
    // connection delay allows: periods 1, 2-3, 4-7, 8-15, and so on. With K the
    // slowest class over all ranks, classes 0 to K-1 become neighbourhoods and
    // class K the far level. So no level is exchanged more often than twice as
    // often as its delays require, and if all delays are equal, all ranks are
    // far ranks, which is the default exchange. The levels are chosen anew
    // whenever the connections change.
    bool automatic = false;
};

// A level of the neighbourhood spike exchange, as seen by the calling rank.
struct spike_exchange_level {
    // Ranks this rank receives spikes from in this level, in ascending order.
    std::vector<int> ranks;
    // Exchanged every `period` epochs; 0 if no connections use this level.
    std::size_t period = 0;
    // Exchanged with the global all-to-all instead of a neighbourhood all-gather.
    bool far = false;
};

} // namespace arb
