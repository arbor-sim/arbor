#pragma once

#ifndef NMC_USE_LINEAR_SPIKE_QUEUE
#error "linear_spike_queue_policy.hpp should only be compiled in a NMC_USE_LINEAR_SPIKE_QUEUE build"
#endif

#include <util/range.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

class linear_spike_event_queue: public base_event_queue {

public:
    // go over all spikes, search for connections with the same source as the spike
    // and then push an event to each connection
    // O(spikes * log (connections) * connections/spike)
    void make_event_queues(const gathered_vector<spike>& global_spikes,
                           std::vector<event_queue>& queues,
                           const std::vector<connection>& connections,
                           const gid_partition_type& cell_gid_partition)
    {
        for (const auto spike: global_spikes.values()) {
            auto targets = std::equal_range(connections.begin(),
                                            connections.end(),
                                            spike.source);

            for (const auto con: make_range(targets)) {
                const auto gidx = cell_group_index(cell_gid_partition, con.destination().gid);
                queues[gidx].push_back(con.make_event(spike));
            }
        }
    }
};

template <typename CommunicationPolicy>
using communicator = base_communicator<CommunicationPolicy, linear_spike_event_queue>;

}
}
}
