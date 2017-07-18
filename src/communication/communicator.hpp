#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <functional>

#include <algorithms.hpp>
#include <common_types.hpp>
#include <connection.hpp>
#include <communication/gathered_vector.hpp>
#include <domain_decomposition.hpp>
#include <event_queue.hpp>
#include <recipe.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/double_buffer.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>

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
    using communication_policy_type = CommunicationPolicy;

    /// per-cell group lists of events to be delivered
    using event_queue =
        std::vector<postsynaptic_spike_event>;

    using gid_partition_type =
        util::partition_range<std::vector<cell_gid_type>::const_iterator>;

    communicator() {}

    explicit communicator(
            const recipe& rec,
            const domain_decomposition& dom_dec,
            const gid_prop_map& gid_props)
    {
        using util::make_span;
        num_domains_ = comms_.size();
        num_local_groups_ = dom_dec.num_local_groups();

        // Make a list of local gid.
        std::vector<cell_gid_type> gids;
        gids.reserve(dom_dec.num_local_cells());
        for (auto i: make_span(0, num_local_groups_)) {
            const auto& group = dom_dec.get_group(i);
            for (auto gid: group.gids()) {
                gids.push_back(gid);
            }
        }

        // Count the number of local connections (i.e. connections terminating on this domain)
        //  -> n_cons: scalar
        // Calculate and store domain id of the presynaptic cell on each local connection
        //  -> ps_doms: array with one entry for every local connection
        // Also the count of presynaptic sources from each domain
        //  -> ps_cnts: array with one entry for each domain
        cell_local_size_type n_cons = 0;
        std::vector<unsigned> ps_doms;
        std::vector<cell_size_type> ps_cnts(num_domains_);
        for (auto gid: gids) {
            const auto conns = rec.connections_on(gid);
            n_cons += conns.size();
            for (auto con: conns) {
                const auto src = dom_dec.gid_domain(con.source.gid);
                ps_doms.push_back(src);
                ps_cnts[src]++;
            }
        }

        // Construct the connections.
        // The loop above gave the information required to construct in place
        // the connections as partitioned by the domain of their source gid.
        connections_.resize(n_cons);
        connection_part_ = algorithms::make_index(ps_cnts);
        auto offsets = connection_part_;
        std::size_t pos = 0;
        for (auto gid: gids) {
            const auto lg = gid_props.get(gid)->local_group;
            for (auto c: rec.connections_on(gid)) {
                const auto i = offsets[ps_doms[pos]]++;
                connections_[i] = {c.source, c.dest, c.weight, c.delay, lg};
                ++pos;
            }
        }

        // Sort the connections for each domain.
        // This is num_domains_ independent sorts, so it can be parallelized trivially.
        threading::parallel_for::apply(0, num_domains_,
            [&](cell_gid_type i) {
                const auto b = connection_part_[i];
                const auto e = connection_part_[i+1];
                util::sort(util::subrange_view(connections_, b, e));
            });
    }

    /// the minimum delay of all connections in the global network.
    time_type min_delay() {
        auto local_min = std::numeric_limits<time_type>::max();
        for (auto& con : connections_) {
            local_min = std::min(local_min, con.delay());
        }

        return comms_.min(local_min);
    }

    /// Perform exchange of spikes.
    ///
    /// Takes as input the list of local_spikes that were generated on the calling domain.
    /// Returns the full global set of vectors, along with meta data about their partition
    gathered_vector<spike> exchange(std::vector<spike> local_spikes) {
        // global all-to-all to gather a local copy of the global spike list on each node.
        util::sort_by(local_spikes, [](spike s){return s.source;});
        auto global_spikes = comms_.gather_spikes(local_spikes);
        num_spikes_ += global_spikes.size();
        return global_spikes;
    }

    /// Check each global spike in turn to see it generates local events.
    /// If so, make the events and insert them into the appropriate event list.
    /// Return a vector that contains the event queues for each local cell group.
    ///
    /// Returns a vector of event queues, with one queue for each local cell group. The
    /// events in each queue are all events that must be delivered to targets in that cell
    /// group as a result of the global spike exchange.
    std::vector<event_queue> make_event_queues(const gathered_vector<spike>& global_spikes) {
        using util::subrange_view;
        using util::make_span;
        using util::make_range;

        auto queues = std::vector<event_queue>(num_local_groups_);
        const auto& sp = global_spikes.partition();
        const auto& cp = connection_part_;
        for (auto dom: make_span(0, num_domains_)) {
            auto cons = subrange_view(connections_, cp[dom], cp[dom+1]);
            auto spks = subrange_view(global_spikes.values(), sp[dom], sp[dom+1]);

            /*
            if (cons.size()<spks.size()) {
                auto sp = spks.begin();
                auto cn = cons.begin();
                while (cn!=cons.end() && sp!=spks.end()) {
                    auto sources = std::equal_range(sp, spks.end(), cn->source());

                    for (auto s: make_range(sources.first, sources.second)) {
                        queues[cn->group_index()].push_back(cn->make_event(s));
                    }

                    sp = sources.first;
                    ++cn;
                }
            }
            else {
            */
                auto cn = cons.begin();
                auto sp = spks.begin();
                while (cn!=cons.end() && sp!=spks.end()) {
                    auto targets = std::equal_range(cn, cons.end(), sp->source);

                    for (auto c: make_range(targets.first, targets.second)) {
                        queues[c.group_index()].push_back(c.make_event(*sp));
                    }

                    cn = targets.first;
                    ++sp;
                }
            //}
        }

        return queues;
    }

    /// Returns the total number of global spikes over the duration of the simulation
    std::uint64_t num_spikes() const { return num_spikes_; }

    const std::vector<connection>& connections() const {
        return connections_;
    }

    void reset() {
        num_spikes_ = 0;
    }

private:
    cell_size_type num_local_groups_;
    cell_size_type num_domains_;
    std::vector<connection> connections_;
    std::vector<cell_size_type> connection_part_;
    communication_policy_type comms_;
    std::uint64_t num_spikes_ = 0u;
};

} // namespace communication
} // namespace mc
} // namespace nest
