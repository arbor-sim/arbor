#pragma once

#include <iostream>

#include <util/range.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;
using nest::mc::util::lessthan;

template <typename CommunicationPolicy>
class domain_search_communicator: public base_communicator<CommunicationPolicy> {
public:
    using base = base_communicator<CommunicationPolicy>;
    using typename base::event_queue;
    using typename base::gid_partition_type;
    using typename base::lt_spike_src;
    
    using base::num_groups_local;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    using base::base;

    // go over each block of connections with the same source, and search for the
    // associated block of spikes with the same source, and then push the product of events
    // O(connection/spike * log(spikes/node))
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes)
    {
        // queues to return
        auto queues = std::vector<event_queue>(num_groups_local());

        // con_next is the beginning of the current block of connections with
        // the same source
        auto con_next = connections_.cbegin();
        const auto con_end = connections_.cend();

        // Search for next block of spikes and connections with the same sender
        while (con_next != con_end) {
            // we grab the next block of connections from the same sender
            const auto src = con_next->source();
            const auto domain = con_next->domain();
            const auto targets = std::equal_range(con_next, con_end, src);
            con_next = targets.second; // next iteration, next conn block

            // we grab the block of spikes (by domain) associated with the connections
            const auto domain_spikes = global_spikes.values_for_partition(domain);
            const auto sources = std::equal_range(domain_spikes.first,
                                                  domain_spikes.second,
                                                  src, lt_spike_src());
            if (sources.first == sources.second) {
                continue; // skip if no spikes == to src
            }

            // Now we just need to walk over all combinations of matching spikes and connections
            // Do it first by connection because of shared data
            for (auto&& con: make_range(targets)) {
                const auto gidx = cell_group_index(con.destination().gid);
                for (auto&& spike: make_range(sources)) {
                   queues[gidx].push_back(con.make_event(spike));
                }
            }
        }

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = domain_search_communicator<CommunicationPolicy>;

}
}
}
