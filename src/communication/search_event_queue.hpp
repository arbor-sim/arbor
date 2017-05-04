#pragma once

#include <util/range.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

// base for class S search_impl in search_event_queues, defining some common
// types
class base_search {
protected:
    using spike_it = std::vector<spike>::const_iterator;
    using con_it = std::vector<connection>::const_iterator;
    using lt_spike_src = base_event_queue::lt_spike_src;
};

template<class S>
class search_event_queue: public base_event_queue {
public:
    // go over each block of connections with the same source, and search for the
    // associated block of spikes with the same source, and then push the product of events
    // O(connections/spike * x) where x is the order for finding the spikes for the block
    // of connections, determined by `S search_impl.get_sources` where the search is done.
    void make_event_queues(const gathered_vector<spike>& global_spikes,
                           std::vector<event_queue>& queues,
                           const std::vector<connection>& connections,
                           const gid_partition_type& cell_gid_partition)
    {
        S search_impl(global_spikes);

        // con_next is the beginning of the current block of connections with
        // the same source
        auto con_next = connections.cbegin();
        const auto con_end = connections.cend();

        // Search for next block of spikes and connections with the same sender
        while (con_next != con_end) {
            // we grab the next block of connections from the same sender
            const auto src = con_next->source();
            const auto sources = search_impl.get_sources(con_next, src);

            const auto con_last = con_next++; // move con_next to end of block
            while (con_next < con_end && con_next->source().gid == src.gid)
                con_next++;

            // skip if no spikes directed to targets
            if (sources.first == sources.second) {
                continue;
            }

            // Now we just need to walk over all combinations of matching spikes and connections
            // Do it first by connection because of shared data
            for (const auto con: make_range(con_last, con_next)) {
                const auto gidx = cell_group_index(cell_gid_partition, con.destination().gid);
                for (const auto spike: make_range(sources)) {
                   queues[gidx].push_back(con.make_event(spike));
                }
            }
        }
    }
};

}
}
}
