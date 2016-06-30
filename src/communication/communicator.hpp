#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

#include <communication/spike.hpp>
#include <threading/threading.hpp>
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
template <typename CommunicationPolicy>
class communicator {
public:
    using id_type = uint32_t;
    using communication_policy_type = CommunicationPolicy;

    using spike_type = spike<id_type>;

    communicator() = default;

    communicator(id_type n_groups, std::vector<id_type> target_counts) :
        num_groups_local_(n_groups),
        num_targets_local_(target_counts.size())
    {
        target_map_ = nest::mc::algorithms::make_index(target_counts);
        num_targets_local_ = target_map_.back();

        // create an event queue for each target group
        events_.resize(num_groups_local_);

        // make maps for converting lid to gid
        target_gid_map_ = communication_policy_.make_map(num_targets_local_);
        group_gid_map_  = communication_policy_.make_map(num_groups_local_);

        // transform the target ids from lid to gid
        auto first_target = target_gid_map_[domain_id()];
        for (auto &id : target_map_) {
            id += first_target;
        }
    }

    id_type target_gid_from_group_lid(id_type lid) const {
        EXPECTS(lid<num_groups_local_);
        return target_map_[lid];
    }

    id_type group_gid_from_group_lid(id_type lid) const {
        EXPECTS(lid<num_groups_local_);
        return group_gid_map_[domain_id()] + lid;
    }

    void add_connection(connection con) {
        EXPECTS(is_local_target(con.destination()));
        connections_.push_back(con);
    }

    bool is_local_target(id_type gid) {
        return gid>=target_gid_map_[domain_id()]
            && gid<target_gid_map_[domain_id()+1];
    }

    bool is_local_group(id_type gid) {
        return gid>=group_gid_map_[domain_id()]
            && gid<group_gid_map_[domain_id()+1];
    }

    /// return the global id of the first group in domain d
    /// the groups in domain d are in the contiguous half open range
    ///     [domain_first_group(d), domain_first_group(d+1))
    id_type group_gid_first(int d) const {
        return group_gid_map_[d];
    }

    id_type target_lid(id_type gid) {
        EXPECTS(is_local_group(gid)); 

        return gid - target_gid_map_[domain_id()];
    }

    id_type group_lid(id_type gid) {
        EXPECTS(is_local_group(gid));

        return gid - group_gid_map_[domain_id()];
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

    // return the local group index of the group which hosts the target with
    // global id gid
    id_type local_group_from_global_target(id_type gid) {
        // assert that gid is in range
        EXPECTS(is_local_target(gid));

        return
            std::distance(
                target_map_.begin(),
                std::upper_bound(target_map_.begin(), target_map_.end(), gid)
            ) - 1;
    }

    void add_spike(spike_type s) {
        thread_spikes().push_back(s);
    }

    void add_spikes(const std::vector<spike_type>& s) {
        auto& v = thread_spikes();
        v.insert(v.end(), s.begin(), s.end());
    }

    std::vector<spike_type>& thread_spikes() {
        return thread_spikes_.local();
    }

    void exchange() {

        // global all-to-all to gather a local copy of the global spike list
        // on each node
        //profiler_.enter("global exchange");
        auto global_spikes = communication_policy_.gather_spikes(local_spikes());
        num_spikes_ += global_spikes.size();
        clear_thread_spike_buffers();
        //profiler_.leave();

        for (auto& q : events_) {
            q.clear();
        }

        //profiler_.enter("events");

        //profiler_.enter("make events");
        // check all global spikes to see if they will generate local events
        for (auto spike : global_spikes) {
            // search for targets
            auto targets =
                std::equal_range(
                    connections_.begin(), connections_.end(), spike.source
                );

            // generate an event for each target
            for (auto it=targets.first; it!=targets.second; ++it) {
                auto gidx = local_group_from_global_target(it->destination());

                events_[gidx].push_back(it->make_event(spike));
            }
        }

        //profiler_.leave(); // make events

        //profiler_.leave(); // event generation
    }

    uint64_t num_spikes() const
    {
        return num_spikes_;
    }

    int domain_id() const {
        return communication_policy_.id();
    }

    int num_domains() const {
        return communication_policy_.size();
    }

    const std::vector<postsynaptic_spike_event>& queue(int i) const {
        return events_[i];
    }

    const std::vector<connection>& connections() const {
        return connections_;
    }

    communication_policy_type communication_policy() const {
        return communication_policy_;
    }

    const std::vector<id_type>& local_target_map() const {
        return target_map_;
    }

    std::vector<spike_type> local_spikes() {
        std::vector<spike_type> spikes;
        for (auto& v : thread_spikes_) {
            spikes.insert(spikes.end(), v.begin(), v.end());
        }
        return spikes;
    }

    void clear_thread_spike_buffers() {
        for (auto& v : thread_spikes_) {
            v.clear();
        }
    }

private:

    //
    //  both of these can be fixed with double buffering
    //
    // FIXME : race condition on the thread_spikes_ buffers when exchange() modifies/access them
    //         ... other threads will be pushing to them simultaneously
    // FIXME : race condition on the group-specific event queues when exchange pushes to them
    //         ... other threads will be accessing them to update their event queues

    // thread private storage for accumulating spikes
    using local_spike_store_type =
        nest::mc::threading::enumerable_thread_specific<std::vector<spike_type>>;
    local_spike_store_type thread_spikes_;

    std::vector<connection> connections_;
    std::vector<std::vector<postsynaptic_spike_event>> events_;

    // local target group i has targets in the half open range
    //      [target_map_[i], target_map_[i+1])
    std::vector<id_type> target_map_;

    // for keeping track of how time is spent where
    //util::Profiler profiler_;

    // the number of groups and targets handled by this communicator
    id_type num_groups_local_;
    id_type num_targets_local_;

    // index maps for the global distribution of groups and targets

    // communicator i has the groups in the half open range :
    //      [group_gid_map_[i], group_gid_map_[i+1])
    std::vector<id_type> group_gid_map_;

    // communicator i has the targets in the half open range :
    //      [target_gid_map_[i], target_gid_map_[i+1])
    std::vector<id_type> target_gid_map_;

    communication_policy_type communication_policy_;

    uint64_t num_spikes_ = 0u;
};

} // namespace communication
} // namespace mc
} // namespace nest
