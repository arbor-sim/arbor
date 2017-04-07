#pragma once

#ifndef NMC_USE_GLOBAL_SEARCH_QUEUE
#error "global_search_queue_policy.hpp should only be compiled in a NMC_USE_GLOBAL_SEARCH_QUEUE build"
#endif

#include <util/range.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

template <typename CommunicationPolicy>
class global_search_communicator: public base_communicator<CommunicationPolicy> {
public:
    using base = base_communicator<CommunicationPolicy>;
    using typename base::event_queue;
    using typename base::lt_spike_src;    
    using base::num_groups_local;
    using base::base;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    // go over each block of connections with the same source, and search for the
    // associated block of spikes with the same source, and then push the product of events
    // O(connections/spike * log(spikes))
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes)
    {
        // queues to return
        auto queues = std::vector<event_queue>(num_groups_local());

        // con_next is the beginning of the current block of connections with
        // the same source
        auto con_next = connections_.cbegin();
        const auto con_end = connections_.cend();

        // spikes_next is the beginning of the current block of spikes with
        // the same source
        const auto& spikes = global_spikes.values();
        auto spikes_next = spikes.cbegin();
        const auto spikes_end = spikes.cend();

        // Search for next block of spikes and connections with the same sender
        while (con_next != con_end) {
            // we grab the next block of connections from the same sender
            const auto src = con_next->source();
            const auto targets = std::equal_range(con_next, con_end, src);
            con_next = targets.second; // next iteration, next conn block
            
            // and the associated block of spikes
            const auto sources = std::equal_range(spikes_next, spikes_end,
                                                  src, lt_spike_src());
            spikes_next = sources.second; //next block that is > src

            if (sources.first == sources.second) {
                continue; // skip if no spikes == to source
            }

            // Now we just need to walk over all combinations of matching spikes and connections
            // Do it first by connection because of shared data
            for (const auto con: make_range(targets)) {
                const auto gidx = cell_group_index(con.destination().gid);
                for (const auto spike: make_range(sources)) {
                    queues[gidx].push_back(con.make_event(spike));
                }
            }
        }

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = global_search_communicator<CommunicationPolicy>;


}
}
}
