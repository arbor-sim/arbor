#pragma once

#include <util/range.hpp>
#include <util/compare.hpp>
#include <profiling/profiler.hpp>

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
    using typename base::cmp_spike;
    
    using base::num_groups_local;
    
protected:
    using base::cell_group_index;
    using base::connections_;

public:
    domain_search_communicator(): base() {}

    explicit domain_search_communicator(gid_partition_type cell_gid_partition):
        base(cell_gid_partition)
    {}

    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes)
    {
        // queues to return
        PE("make-queues");
        auto queues = std::vector<event_queue>(num_groups_local());
        PL();

        auto con_next = connections_.cbegin();
        const auto con_end = connections_.cend();

        // For block of connections, search for block of spikes from
        // that sender
        PE("connections");
        while (con_next != con_end) {
            // we grab the next block of connections from the same sender
            const auto src = con_next->source();
            PE("targets");
            const auto targets = std::equal_range(con_next, con_end, src);
            con_next = targets.second; // next iteration, next conn block
            PL();

            PE("spikes");
            // we grab the block of spikes associated with the connections
            const auto domain = con_next->domain();
            const auto domain_spikes = global_spikes.values_for_partition(domain);
            const auto sources = std::equal_range(domain_spikes.first,
                                                  domain_spikes.second,
                                                  src, cmp_spike());
            PL();
            if (sources.first == sources.second) {
                continue; // skip if no spikes
            }

            PE("queue");
            // Now we just need to walk over all combinations of matching spikes and connections
            // Do it first by connection because of shared data
            for (auto&& con: make_range(targets)) {
                const auto gidx = cell_group_index(con.destination().gid);
                auto& queue = queues[gidx];

                for (auto&& spike: make_range(sources)) {
                    queue.push_back(con.make_event(spike));
                }
            }
            PL();
        }
        PL();

        return queues;
    }
};

template <typename CommunicationPolicy>
using communicator = domain_search_communicator<CommunicationPolicy>;

}
}
}
