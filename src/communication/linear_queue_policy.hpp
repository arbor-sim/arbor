#pragma once

#include <util/compare.hpp>

#include "base_communicator.hpp"

namespace nest {
namespace mc {
namespace communication {

using nest::mc::util::make_range;

template <typename CommunicationPolicy>
class linear_communicator: public base_communicator<CommunicationPolicy> {
public:
    using base = base_communicator<CommunicationPolicy>;
    using typename base::event_queue;
    using typename base::gid_partition_type;
    
    using base::num_groups_local;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    linear_communicator(): base() {}

    explicit linear_communicator(gid_partition_type cell_gid_partition):
        base(cell_gid_partition)
    {}

    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes) {
        auto queues = std::vector<event_queue>(num_groups_local());
        for (auto spike : global_spikes.values()) {
            // search for targets
            auto targets = std::equal_range(connections_.begin(), connections_.end(), spike.source);

            // generate an event for each target
            for (auto&& con: make_range(targets)) {
                auto gidx = cell_group_index(con.destination().gid);
                queues[gidx].push_back(con.make_event(spike));
            }
        }

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = linear_communicator<CommunicationPolicy>;

}
}
}
