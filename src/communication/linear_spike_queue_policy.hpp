#pragma once

#include <iostream>

#include <util/compare.hpp>
#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

template <typename CommunicationPolicy>
class linear_spike_communicator: public base_communicator<CommunicationPolicy> {
public:
    using base = base_communicator<CommunicationPolicy>;
    using typename base::event_queue;
    using base::num_groups_local;
    using base::base;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    // go over all spikes, search for connections with the same source as the spike
    // and then push an event to each connection
    // O(spikes * log (connections) * connections/spike)
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes)
    {
        auto queues = std::vector<event_queue>(num_groups_local());
        
        for (auto spike: global_spikes.values()) {
            auto targets = std::equal_range(connections_.begin(),
                                            connections_.end(),
                                            spike.source);

            for (auto&& con: make_range(targets)) {
                const auto gidx = cell_group_index(con.destination().gid);
                queues[gidx].push_back(con.make_event(spike));
            }
        }

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = linear_spike_communicator<CommunicationPolicy>;

}
}
}
