#pragma once

#ifndef NMC_USE_GLOBAL_SEARCH_QUEUE
#error "global_search_queue_policy.hpp should only be compiled in a NMC_USE_GLOBAL_SEARCH_QUEUE build"
#endif

#include "search_event_queue.hpp"

namespace nest {
namespace mc {
namespace communication {

class global_search: public base_search {
public:
    global_search(const gathered_vector<spike>& global_spikes):
        spikes_next_(global_spikes.values().cbegin()),
        spikes_end_(global_spikes.values().cend())
    {}

    // go over each block of connections with the same source, and search for the
    // associated block of spikes with the same source, and then push the product of events
    // O(connections/spike * log(spikes))
    std::pair<spike_it, spike_it> get_sources(const con_it& con_next,
                                              const cell_member_type& src)
    {
        // and the associated block of spikes
        const auto sources = std::equal_range(spikes_next_, spikes_end_,
                                              src, lt_spike_src());
        spikes_next_ = sources.second; //next block that is > src
        return sources;
    }

private:
    // spikes_next is the beginning of the current block of spikes with
    // the same source
    spike_it spikes_next_;
    const spike_it spikes_end_;
};

template <typename CommunicationPolicy>
using communicator = base_communicator<CommunicationPolicy, search_event_queue<global_search>>;

}
}
}
