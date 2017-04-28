#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

#include <algorithms.hpp>
#include <connection.hpp>
#include <communication/gathered_vector.hpp>
#include <event_queue.hpp>
#include <spike.hpp>
#include <util/compare.hpp>
#include <util/debug.hpp>
#include <util/double_buffer.hpp>
#include <util/partition.hpp>

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

// base class for mix-in implementation of make_event_queues
// contains some common types and functions needed by make_event_queues
class base_event_queue {
public:
    /// per-cell group lists of events to be delivered
    using event_queue =
        std::vector<postsynaptic_spike_event>;

    using gid_partition_type =
        util::partition_range<std::vector<cell_gid_type>::const_iterator>;

    struct spike_extractor {
        using id_type = cell_gid_type;
        id_type operator()(const cell_member_type& s) const {return s.gid;}
        id_type operator()(const spike& s) const {return s.source.gid;}
    };

    template<typename E>
    using lessthan = nest::mc::util::lessthan<E>;
    using lt_spike_src = lessthan<spike_extractor>;
    
    static std::size_t cell_group_index(gid_partition_type cell_gid_partition,
                                        cell_gid_type cell_gid)
    {
        EXPECTS(is_local_cell(cell_gid_partition, cell_gid));
        return cell_gid_partition.index(cell_gid);
    }

    static     /// returns true if the cell with gid is on the domain of the caller
    bool is_local_cell(gid_partition_type cell_gid_partition,
                       cell_gid_type gid)
    {
        return algorithms::in_interval(gid, cell_gid_partition.bounds());
    }
};

template <typename CommunicationPolicy, typename EventQueueImpl>
class base_communicator {
public:
    using communication_policy_type = CommunicationPolicy;
    
    using queue_impl = EventQueueImpl;
    queue_impl queue_impl_;

    using event_queue = base_event_queue::event_queue;
    using gid_partition_type = base_event_queue::gid_partition_type;

    explicit base_communicator(gid_partition_type cell_gid_partition):
        cell_gid_partition_(cell_gid_partition)
    {}

    cell_local_size_type num_groups_local() const
    {
        return cell_gid_partition_.size();
    }

    void add_connection(connection con) {
        EXPECTS(is_local_cell(con.destination().gid));
        connections_.push_back(con);
    }

    /// returns true if the cell with gid is on the domain of the caller
    bool is_local_cell(cell_gid_type gid) const {
        return base_event_queue::is_local_cell(cell_gid_partition_, gid);
    }

    /// builds the optimized data structure
    /// must be called after all connections have been added
    void construct() {
        if (!std::is_sorted(connections_.begin(), connections_.end())) {
            threading::sort(connections_);
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
    /// Returns the full global set of vectors, along with meta data about their partition
    gathered_vector<spike> exchange(const std::vector<spike>& local_spikes) {
        // global all-to-all to gather a local copy of the global spike list on each node.
        auto global_spikes = communication_policy_.gather_spikes( local_spikes );
        num_spikes_ += global_spikes.size();
        return global_spikes;
    }

    /// make_event_queues: functions as a mixin in on EvenQueueImpl
    /// setting up the queues to be returned, the work gets done in
    /// EventQueueImpl.make_event_queues, and the queues are returned
    ///
    /// Maps spike events to local connections and inserts the events
    /// in the event queue. Details for searches for relevant spike/connection pairs
    /// are in EventQueueImpl
    ///
    /// Returns a vector of event queues, with one queue for each local cell group. The
    /// events in each queue are all events that must be delivered to targets in that cell
    /// group as a result of the global spike exchange.
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes) {
        // queues to return
        auto queues = std::vector<event_queue>(num_groups_local());
        queue_impl_.make_event_queues(global_spikes, queues, connections_, cell_gid_partition_);
        return queues;
    }

    /// Returns the total number of global spikes over the duration of the simulation
    uint64_t num_spikes() const { return num_spikes_; }

    const std::vector<connection>& connections() const {
        return connections_;
    }

    communication_policy_type communication_policy() const {
        return communication_policy_;
    }

    void reset() {
        num_spikes_ = 0;
    }

protected:
    
    std::vector<connection> connections_;

    communication_policy_type communication_policy_;

    uint64_t num_spikes_ = 0u;

    gid_partition_type cell_gid_partition_;
};

} // namespace communication
} // namespace mc
} // namespace nest
