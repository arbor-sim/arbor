#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>

namespace arb {

// Delay-aware spike exchange via MPI neighbourhood collectives, set for all
// simulations on a context with make_context.
//
// By default, all spikes are exchanged after every epoch, where an epoch is
// half the minimum delay of the whole network. With this option, the ranks that
// the calling rank receives spikes from are split into levels instead:
//
//   * the calling rank itself, which is handled without communication,
//   * one level per neighbourhood, exchanged with a sparse MPI neighbourhood
//     all-gather over just those ranks,
//   * all remaining ("far") ranks, exchanged with a global collective that
//     sends each spike only to the ranks with connections from its source.
//
// Spikes are buffered per level and only sent as often as the minimum delay of
// the connections from that level permits. If, for example, connections between
// neighbouring ranks are much shorter than connections between distant ones,
// the expensive global collective runs only every few epochs. All buffers are
// flushed at the end of each call to simulation::run, so the global spike count
// is exact after run returns.
//
// Every rank must pass the same kind of exchange: off, automatic, or the same
// number of neighbourhoods. This is checked when a simulation is built.
class neighborhood_spike_exchange {
public:
    neighborhood_spike_exchange() = delete;

    // Ranks this rank receives spikes from, grouped into levels, innermost
    // first. The levels must be disjoint: no rank may be listed more than once,
    // in the same level or in different ones. The calling rank is ignored.
    //
    // Without any levels, all other ranks are far ranks: only the delay-aware
    // buffering is used, which still saves collectives if the delays between
    // ranks exceed the minimum delay.
    explicit neighborhood_spike_exchange(std::vector<std::vector<int>> neighborhoods):
        neighborhood_spike_exchange(mode::given, std::move(neighborhoods)) {}

    // The default exchange after every epoch.
    static neighborhood_spike_exchange off() { return {mode::off, {}}; }

    // Choose the levels from the connection delays.
    //
    // Every source rank goes into a class by the exchange period its minimum
    // connection delay allows: periods 1, 2-3, 4-7, 8-15, and so on. With K the
    // slowest class over all ranks, classes 0 to K-1 become neighbourhoods and
    // class K the far level. So no level is exchanged more often than twice as
    // often as its delays require, and if all delays are equal, all ranks are
    // far ranks. The levels are chosen anew whenever the connections change;
    // simulation::spike_exchange_levels returns those in use.
    static neighborhood_spike_exchange automatic() { return {mode::automatic, {}}; }

    bool is_off() const { return mode_ == mode::off; }
    bool is_automatic() const { return mode_ == mode::automatic; }

    // The given neighbourhoods; empty if off or automatic.
    const std::vector<std::vector<int>>& neighborhoods() const { return neighborhoods_; }

private:
    enum class mode { off, given, automatic };

    // Two arguments, so that neighborhood_spike_exchange{{}} means no neighbourhoods.
    neighborhood_spike_exchange(mode m, std::vector<std::vector<int>> neighborhoods):
        mode_(m), neighborhoods_(std::move(neighborhoods)) {}

    mode mode_;
    std::vector<std::vector<int>> neighborhoods_;
};

// A level of the neighbourhood spike exchange, as seen by the calling rank.
struct spike_exchange_level {
    // Ranks this rank receives spikes from in this level, in ascending order.
    std::vector<int> ranks;
    // Exchanged every `period` epochs; 0 if no connections use this level.
    std::size_t period = 0;
    // Minimum delay [ms] of the connections in this level over all ranks, from
    // which the period follows; infinity if there are none.
    time_type min_delay = std::numeric_limits<time_type>::infinity();
    // Exchanged with the global collective instead of a neighbourhood all-gather.
    bool far = false;
};

} // namespace arb
