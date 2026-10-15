#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_exchange.hpp>

#include "communication/gathered_vector.hpp"
#include "distributed_context.hpp"

namespace arb {

// Number of epochs for which spikes can be held back before they must be
// exchanged, given the minimum delay of the connections they travel over and
// the maximum epoch length.
//
// Spikes of epoch e that are exchanged after epoch e+p-1 at the latest become
// events in epoch e+p+1, which starts at most (p+1)*interval after the spike.
// These events are on time if (p+1)*interval <= delay.
ARB_ARBOR_API std::size_t exchange_period(time_type delay, time_type interval);

// Class of an exchange period for choosing levels automatically: floor(log2(period)).
ARB_ARBOR_API unsigned period_class(std::size_t period);

// Sort spikes by source, given that they consist of runs that are already
// sorted by source, starting at the given offsets. Merges the runs pairwise in
// log2(runs) passes, which keeps spikes from the same source in run order.
ARB_ARBOR_API void merge_sorted_runs(std::vector<spike>& spikes, std::vector<std::size_t> runs);

// Implements neighborhood_spike_exchange.
//
// Every rank sorts the ranks it receives from into levels: itself, its
// neighbourhoods, and the far level with all others. The neighbourhoods are
// either given, or chosen from the connection delays on every update. Each source rank belongs
// to exactly one level of each receiver, so every spike reaches every receiver
// exactly once. Spikes are buffered per level and each level is exchanged
// every `period` epochs, where period is computed from the global minimum
// delay of the connections from that level.
//
// All public member functions except levels() are collective.
class ARB_ARBOR_API neighborhood_router {
public:
    using source_ranks_map = std::unordered_map<cell_member_type, std::vector<cell_size_type>>;

    neighborhood_router(const neighborhood_spike_exchange& options, distributed_context_handle dist);

    // Recompute routes and exchange periods after the connections changed.
    //  * min_delays: per domain, the minimum delay of the local connections from it,
    //                infinity if there are none.
    //  * src_ranks:  per local source, the domains with connections from it.
    //  * interval:   maximum epoch length.
    void update(const std::vector<time_type>& min_delays, const source_ranks_map& src_ranks, time_type interval);

    // Buffer the local spikes of epoch epoch_id, which must be sorted by source,
    // and exchange all levels that are due in it. Without an epoch, all levels
    // are exchanged. Returns the spikes received, partitioned by source domain.
    gathered_vector<spike> exchange(const std::vector<spike>& local_spikes, std::optional<std::ptrdiff_t> epoch_id);

    // The levels of this rank: neighbourhoods first, far last.
    std::vector<spike_exchange_level> levels() const;

    // Drop all buffered spikes.
    void clear();

private:
    static constexpr int self_level = -1;

    struct level {
        // Null for the far level.
        distributed_neighborhood_ptr neighbors;
        std::size_t period = 0;
        // Spikes held back, one run sorted by source per epoch, starting at the offsets in runs.
        std::vector<spike> pending;
        std::vector<std::size_t> runs;

        void clear() {
            pending.clear();
            runs.clear();
        }
    };

    // Levels a local source's spikes are sent in, and the ranks it is sent to.
    struct route {
        std::vector<int> levels;
        std::vector<cell_size_type> ranks;
    };

    int far_level() const { return levels_.size() - 1; }
    // Make the neighbourhoods, which must be valid, and assign all other ranks to the far level.
    void set_levels(const std::vector<std::vector<int>>& neighborhoods);
    // Neighbourhoods by period class; see neighborhood_spike_exchange::automatic.
    std::vector<std::vector<int>> automatic_levels(const std::vector<time_type>& min_delays, time_type interval) const;
    // Sort spikes into bins, one per far rank they are sent to.
    gathered_vector<spike> bin_by_far_rank(const std::vector<spike>& spikes) const;

    distributed_context_handle dist_;
    bool automatic_;
    int rank_;
    int num_domains_;
    // Neighbourhoods, then the far level.
    std::vector<level> levels_;
    // Per domain: level in which we receive from it, and in which we send to it.
    std::vector<int> recv_level_;
    std::vector<int> send_level_;
    std::unordered_map<cell_member_type, route> routes_;
};

} // namespace arb
