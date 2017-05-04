#pragma once

#ifndef NMC_USE_DOMAIN_SEARCH_QUEUE
#error "domain_search_queue_policy.hpp should only be compiled in a NMC_USE_DOMAIN_SEARCH_QUEUE build"
#endif

#include "search_event_queue.hpp"

namespace nest {
namespace mc {
namespace communication {

class domain_search: public base_search {
public:
    domain_search(const gathered_vector<spike>& global_spikes):
        global_spikes_(global_spikes)
    {}
    
    // go over each block of connections with the same source, and search for the
    // associated block of spikes with the same source, and then push the product of events
    // O(connection/spike * log(spikes/node))
    std::pair<spike_it, spike_it> get_sources(const con_it& con_next,
                                              const cell_member_type& src)
    {
        const auto domain = con_next->domain();
        const auto domain_spikes = global_spikes_.values_for_partition(domain);        

        // we grab the block of spikes (by domain) associated with the connections
        return std::equal_range(domain_spikes.first,
                                domain_spikes.second,
                                src, lt_spike_src());
    }

private:
    const gathered_vector<spike>& global_spikes_;
};

template <typename CommunicationPolicy>
using communicator = base_communicator<CommunicationPolicy, search_event_queue<domain_search>>;

}
}
}
