#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include <spike.hpp>
#include <threading/threading.hpp>
#include <util/double_buffer.hpp>
#include <algorithms.hpp>
#include <event_queue.hpp>

#include "connection.hpp"

namespace nest {
namespace mc {
namespace communication {

// When the communicator is constructed the number of target groups and targets
// is specified, along with a mapping between local cell id and local
// target id.
//
// The user can add connections to an existing communicator object, where
// each connection is between any global cell and any local target.
//
// Once all connections have been specified, the construct() method can be used
// to build the data structures required for efficient spike communication and
// event generation.
//
// To overlap communication and computation, i.e. to perform spike
// exchange at the same time as cell state update, thread safe access to the
// spike and event lists must be provided. We use double buffering, whereby
// for each of the spikes and events one buffer is exposed publicly, while
// the other is used internally by the communicator 
//  - the spike lists are not directly exposed via the communicator
//    interface. Instead they are updated when the add_spike() methods
//    are called.
//  - the event queues for each cell group are exposed via the queue()
//    method.
// For each double buffer, the current buffer (accessed via buffer.get())
// is exposed to the user, and the other buffer is used inside exchange().

template <typename CommunicationPolicy>
class communicator {
private:
    using communication_policy_type = CommunicationPolicy;
    using id_type = cell_gid_type;
    using spike_type = spike<cell_member_type>;

    /// thread private storage for accumulating spikes
    using local_spike_store_type =
        threading::enumerable_thread_specific<std::vector<spike_type>>;

    /// per-cell group lists of events to be delivered
    using event_queue =
        std::vector<postsynaptic_spike_event>;

    /// double buffered storage of the thread private spike lists
    util::double_buffer<local_spike_store_type> thread_spikes_;

    /// double buffered storage of the cell group event lists
    util::double_buffer<std::vector<event_queue>> events_;

    /// access to the spikes buffered from the previous communication
    /// interval. Used internally by the communicator for exchange
    local_spike_store_type& buffered_spikes() {
        return thread_spikes_.other();
    }

    /// access to thread-private list of spikes, used for storing
    /// spikes added via the add_spike() interface
    std::vector<spike_type>& thread_spikes() {
        return thread_spikes_.get().local();
    }

    void clear_buffered_spikes() {
        for (auto& v : buffered_spikes()) {
            v.clear();
        }
    }

    std::vector<spike_type> gather_local_spikes() {
        std::vector<spike_type> spikes;
        for (auto& v : buffered_spikes()) {
            spikes.insert(spikes.end(), v.begin(), v.end());
        }
        return spikes;
    }

    std::vector<connection> connections_;

    communication_policy_type communication_policy_;

    uint64_t num_spikes_ = 0u;
    id_type cell_gid_from_;
    id_type cell_gid_to_;

public:
    communicator() = default;

    // for now, still assuming one-to-one association cells <-> groups,
    // so that 'group' gids as represented by their first cell gid are
    // contiguous.
    communicator(id_type cell_from, id_type cell_to):
        cell_gid_from_(cell_from), cell_gid_to_(cell_to)
    {
        auto num_groups_local_ = cell_gid_to_-cell_gid_from_;

        // create an event queue for each target group
        events_.get().resize(num_groups_local_);
        events_.other().resize(num_groups_local_);
    }

    void add_connection(connection con) {
        EXPECTS(is_local_cell(con.destination().gid));
        connections_.push_back(con);
    }

    bool is_local_cell(id_type gid) const {
        return gid>=cell_gid_from_ && gid<cell_gid_to_;
    }

    // builds the optimized data structure
    void construct() {
        if (!std::is_sorted(connections_.begin(), connections_.end())) {
            std::sort(connections_.begin(), connections_.end());
        }
    }

    float min_delay() {
        auto local_min = std::numeric_limits<float>::max();
        for (auto& con : connections_) {
            local_min = std::min(local_min, con.delay());
        }

        return communication_policy_.min(local_min);
    }

    void add_spike(spike_type s) {
        thread_spikes().push_back(s);
    }

    void add_spikes(const std::vector<spike_type>& s) {
        auto& v = thread_spikes();
        v.insert(v.end(), s.begin(), s.end());
    }

    void exchange() {
        // global all-to-all to gather a local copy of the global spike list
        // on each node
        auto global_spikes = communication_policy_.gather_spikes(gather_local_spikes());
        num_spikes_ += global_spikes.size();
        clear_buffered_spikes();

        // clear the event queue buffers, which will hold the events generated by the
        // global_spikes in the exchange
        auto& queues = events_.other();
        for (auto& q : queues) {
            q.clear();
        }

        // check all global spikes to see if they will generate local events
        for (auto spike : global_spikes) {
            // search for targets
            auto targets =
                std::equal_range(
                    connections_.begin(), connections_.end(), spike.source
                );

            // generate an event for each target
            for (auto it=targets.first; it!=targets.second; ++it) {
                auto gidx = it->destination().gid - cell_gid_from_;
                queues[gidx].push_back(it->make_event(spike));
            }
        }
    }

    uint64_t num_spikes() const { return num_spikes_; }

    const std::vector<postsynaptic_spike_event>& queue(int i) const {
        return events_.get()[i];
    }

    const std::vector<connection>& connections() const {
        return connections_;
    }

    communication_policy_type communication_policy() const {
        return communication_policy_;
    }

    void swap_buffers() {
        thread_spikes_.exchange();
        events_.exchange();
    }

};

} // namespace communication
} // namespace mc
} // namespace nest
