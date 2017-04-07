#pragma once

#ifndef NMC_USE_LINEAR_CONNECTION_QUEUE
#error "linear_connection_queue_policy.hpp should only be compiled in a NMC_USE_LINEAR_CONNECTION_QUEUE build"
#endif

#include <util/compare.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

template <typename CommunicationPolicy>
class linear_connection_communicator: public base_communicator<CommunicationPolicy> {
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
    // go over all connections, search for spikes with the same source as the connection
    // and push an event on the connection for each of those spikes
    // O(connections/node * log(spikes) * spikes/connection)
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes)
    {
        auto queues = std::vector<event_queue>(num_groups_local());

        const auto& spikes = global_spikes.values();
        const auto spikes_begin = spikes.cbegin();
        const auto spikes_end = spikes.cend();

        for (const auto con: connections_) {
            const auto gidx = cell_group_index(con.destination().gid);
            // search for spikes for this connection
            auto con_spikes = std::equal_range(spikes_begin, spikes_end,
                                               con.source(), lt_spike_src());

            for (const auto spike: make_range(con_spikes)) {
                queues[gidx].push_back(con.make_event(spike));
            }
        }

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = linear_connection_communicator<CommunicationPolicy>;

}
}
}
