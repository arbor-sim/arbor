#pragma once

#ifndef NMC_USE_LINEAR_CONNECTION_QUEUE
#error "linear_connection_queue_policy.hpp should only be compiled in a NMC_USE_LINEAR_CONNECTION_QUEUE build"
#endif

#include <util/range.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

class linear_connection_event_queue: public base_event_queue {    
public:
    // go over all connections, search for spikes with the same source as the connection
    // and push an event on the connection for each of those spikes
    // O(connections/node * log(spikes) * spikes/connection)
    void make_event_queues(const gathered_vector<spike>& global_spikes,
                           std::vector<event_queue>& queues,
                           const std::vector<connection>& connections,
                           const gid_partition_type& cell_gid_partition)
    {
        const auto& spikes = global_spikes.values();
        const auto spikes_begin = spikes.cbegin();
        const auto spikes_end = spikes.cend();

        for (const auto con: connections) {
            const auto gidx = cell_group_index(cell_gid_partition, con.destination().gid);
            // search for spikes for this connection
            auto con_spikes = std::equal_range(spikes_begin, spikes_end,
                                               con.source(), lt_spike_src());

            for (const auto spike: make_range(con_spikes)) {
                queues[gidx].push_back(con.make_event(spike));
            }
        }
    }
};

template <typename CommunicationPolicy>
using communicator = base_communicator<CommunicationPolicy, linear_connection_event_queue>;

}
}
}
