#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <functional>

#include <spike.hpp>
#include <util/double_buffer.hpp>
#include <algorithms.hpp>
#include <connection.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <util/debug.hpp>

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
template <typename Time, typename CommunicationPolicy>
class communicator {
public:
    using communication_policy_type = CommunicationPolicy;
    using id_type = cell_gid_type;
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    using connection_type = connection<time_type>;

    /// per-cell group lists of events to be delivered
    using event_queue =
        std::vector<postsynaptic_spike_event<time_type>>;

    communicator() = default;

    // TODO
    // for now, still assuming one-to-one association cells <-> groups,
    // so that 'group' gids as represented by their first cell gid are
    // contiguous.
    communicator(id_type cell_from, id_type cell_to):
        cell_gid_from_(cell_from), cell_gid_to_(cell_to)
    {}

    cell_local_size_type num_groups_local() const
    {
        return cell_gid_to_-cell_gid_from_;
    }

    void add_connection(connection_type con) {
        EXPECTS(is_local_cell(con.destination().gid));
        connections_.push_back(con);
    }

    /// returns true if the cell with gid is on the domain of the caller
    bool is_local_cell(id_type gid) const {
        return gid>=cell_gid_from_ && gid<cell_gid_to_;
    }

    /// builds the optimized data structure
    /// must be called after all connections have been added
    void construct() {
        if (!std::is_sorted(connections_.begin(), connections_.end())) {
            std::sort(connections_.begin(), connections_.end());
        }
    }

    /// the minimum delay of all connections in the global network.
    time_type min_delay() {
        auto local_min = std::numeric_limits<time_type>::max();
        for (auto& con : connections_) {
            local_min = std::min(local_min, con.delay());
        }

        return communication_policy_.min(local_min);
    }

    /// Perform exchange of spikes.
    ///
    /// Takes as input the list of local_spikes that were generated on the calling domain.
    ///
    /// Returns a vector of event queues, with one queue for each local cell group. The
    /// events in each queue are all events that must be delivered to targets in that cell
    /// group as a result of the global spike exchange.
    std::vector<event_queue> exchange(const std::vector<spike_type>& local_spikes,
        std::function<void ()> export_function) //const std::vector<spike_type>&
    {
        // global all-to-all to gather a local copy of the global spike list on each node.
        
        bool file_per_rank = true;
        if (file_per_rank) {
            export_function(); //local_spikes
        }


        auto global_spikes = communication_policy_.gather_spikes( local_spikes );
        num_spikes_ += global_spikes.size();

        // check each global spike in turn to see it generates local events.
        // if so, make the events and insert them into the appropriate event list.
        auto queues = std::vector<event_queue>(num_groups_local());

        for (auto spike : global_spikes) {
            // search for targets
            auto targets =
                std::equal_range(
                    connections_.begin(), connections_.end(), spike.source
                );

            // generate an event for each target
            for (auto it=targets.first; it!=targets.second; ++it) {
                auto gidx = cell_group_index(it->destination().gid);
                queues[gidx].push_back(it->make_event(spike));
            }
        }

        return queues;
    }

    /// Returns the total number of global spikes over the duration
    /// of the simulation
    uint64_t num_spikes() const { return num_spikes_; }

    const std::vector<connection_type>& connections() const {
        return connections_;
    }

    communication_policy_type communication_policy() const {
        return communication_policy_;
    }

private:
    std::size_t cell_group_index(cell_gid_type cell_gid) const {
        // this will be more elaborate when there is more than one cell per cell group
        EXPECTS(cell_gid>=cell_gid_from_ && cell_gid<cell_gid_to_);
        return cell_gid-cell_gid_from_;
    }

    std::vector<connection_type> connections_;

    communication_policy_type communication_policy_;

    uint64_t num_spikes_ = 0u;
    id_type cell_gid_from_;
    id_type cell_gid_to_;
};

} // namespace communication
} // namespace mc
} // namespace nest
