#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <algorithms.hpp>
#include <common_types.hpp>
#include <communication/gathered_vector.hpp>
#include <connection.hpp>
#include <domain_decomposition.hpp>
#include <event_queue.hpp>
#include <recipe.hpp>
#include <spike.hpp>
#include <util/debug.hpp>
#include <util/double_buffer.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>

namespace arb {
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

    communicator() {}

    explicit communicator(const recipe& rec, const domain_decomposition& dom_dec) {
        using util::make_span;
        num_domains_ = comms_.size();
        num_local_groups_ = dom_dec.groups.size();

        // For caching information about each cell
        struct gid_info {
            using connection_list = decltype(std::declval<recipe>().connections_on(0));
            cell_gid_type gid;              // global identifier of cell
            cell_size_type index_on_domain; // index of cell in this domain
            connection_list conns;          // list of connections terminating at this cell
            gid_info(cell_gid_type g, cell_size_type di, connection_list c):
                gid(g), index_on_domain(di), conns(std::move(c)) {}
        };

        // Make a list of local gid with their group index and connections
        //   -> gid_infos
        // Count the number of local connections (i.e. connections terminating on this domain)
        //   -> n_cons: scalar
        // Calculate and store domain id of the presynaptic cell on each local connection
        //   -> src_domains: array with one entry for every local connection
        // Also the count of presynaptic sources from each domain
        //   -> src_counts: array with one entry for each domain
        std::vector<gid_info> gid_infos;
        gid_infos.reserve(dom_dec.num_local_cells);

        cell_local_size_type n_cons = 0;
        cell_local_size_type n_gid = 0;
        std::vector<unsigned> src_domains;
        std::vector<cell_size_type> src_counts(num_domains_);
        for (auto g: make_span(0, num_local_groups_)) {
            const auto& group = dom_dec.groups[g];
            for (auto gid: group.gids) {
                gid_info info(gid, n_gid, rec.connections_on(gid));
                n_cons += info.conns.size();
                for (auto con: info.conns) {
                    const auto src = dom_dec.gid_domain(con.source.gid);
                    src_domains.push_back(src);
                    src_counts[src]++;
                }
                gid_infos.push_back(std::move(info));
                ++n_gid;
            }
        }

        num_local_cells_ = n_gid;

        // Construct the connections.
        // The loop above gave the information required to construct in place
        // the connections as partitioned by the domain of their source gid.
        connections_.resize(n_cons);
        connection_part_ = algorithms::make_index(src_counts);
        auto offsets = connection_part_;
        std::size_t pos = 0;
        for (const auto& cell: gid_infos) {
            for (auto c: cell.conns) {
                const auto i = offsets[src_domains[pos]]++;
                connections_[i] = {c.source, c.dest, c.weight, c.delay, cell.index_on_domain};
                ++pos;
            }
        }

        // Build cell partition by group for passing events to cell groups
        index_part_ = util::make_partition(index_divisions_,
            util::transform_view(
                dom_dec.groups,
                [](const group_description& g){return g.gids.size();}));

        // Sort the connections for each domain.
        // This is num_domains_ independent sorts, so it can be parallelized trivially.
        const auto& cp = connection_part_;
        threading::parallel_for::apply(0, num_domains_,
            [&](cell_size_type i) {
                util::sort(util::subrange_view(connections_, cp[i], cp[i+1]));
            });
    }

    /// The range of event queues that belong to cells in group i.
    std::pair<cell_size_type, cell_size_type> group_queue_range(cell_size_type i) {
        EXPECTS(i<num_local_groups_);
        return index_part_[i];
    }

    /// The minimum delay of all connections in the global network.
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
        // sort the spikes in ascending order of source gid
        util::sort_by(local_spikes, [](spike s){return s.source;});

        // global all-to-all to gather a local copy of the global spike list on each node.
        auto global_spikes = comms_.gather_spikes(local_spikes);
        num_spikes_ += global_spikes.size();
        return global_spikes;
    }

    /// Check each global spike in turn to see it generates local events.
    /// If so, make the events and insert them into the appropriate event list.
    ///
    /// Takes reference to a vector of event lists as an argument, with one list
    /// for each local cell group. On completion, the events in each list are
    /// all events that must be delivered to targets in that cell group as a
    /// result of the global spike exchange, plus any events that were already
    /// in the list.
    void make_event_queues(
            const gathered_vector<spike>& global_spikes,
            std::vector<pse_vector>& queues)
    {
        EXPECTS(queues.size()==num_local_cells_);

        using util::subrange_view;
        using util::make_span;
        using util::make_range;

        const auto& sp = global_spikes.partition();
        const auto& cp = connection_part_;
        for (auto dom: make_span(0, num_domains_)) {
            auto cons = subrange_view(connections_, cp[dom], cp[dom+1]);
            auto spks = subrange_view(global_spikes.values(), sp[dom], sp[dom+1]);

            struct spike_pred {
                bool operator()(const spike& spk, const cell_member_type& src)
                    {return spk.source<src;}
                bool operator()(const cell_member_type& src, const spike& spk)
                    {return src<spk.source;}
            };

            // We have a choice of whether to walk spikes or connections:
            // i.e., we can iterate over the spikes, and for each spike search
            // the for connections that have the same source; or alternatively
            // for each connection, we can search the list of spikes for spikes
            // with the same source.
            //
            // We iterate over whichever set is the smallest, which has
            // complexity of order max(S log(C), C log(S)), where S is the
            // number of spikes, and C is the number of connections.
            if (cons.size()<spks.size()) {
                auto sp = spks.begin();
                auto cn = cons.begin();
                while (cn!=cons.end() && sp!=spks.end()) {
                    auto sources = std::equal_range(sp, spks.end(), cn->source(), spike_pred());
                    for (auto s: make_range(sources)) {
                        queues[cn->index_on_domain()].push_back(cn->make_event(s));
                    }

                    sp = sources.first;
                    ++cn;
                }
            }
            else {
                auto cn = cons.begin();
                auto sp = spks.begin();
                while (cn!=cons.end() && sp!=spks.end()) {
                    auto targets = std::equal_range(cn, cons.end(), sp->source);
                    for (auto c: make_range(targets)) {
                        queues[c.index_on_domain()].push_back(c.make_event(*sp));
                    }

                    cn = targets.first;
                    ++sp;
                }
            }
        }
    }

    /// Returns the total number of global spikes over the duration of the simulation
    std::uint64_t num_spikes() const { return num_spikes_; }

    cell_size_type num_local_cells() const {
        return num_local_cells_;
    }

    const std::vector<connection>& connections() const {
        return connections_;
    }

    void reset() {
        num_spikes_ = 0;
    }

private:
    cell_size_type num_local_cells_;
    cell_size_type num_local_groups_;
    cell_size_type num_domains_;
    std::vector<connection> connections_;
    std::vector<cell_size_type> connection_part_;
    std::vector<cell_size_type> index_divisions_;
    util::partition_view_type<std::vector<cell_size_type>> index_part_;

    communication_policy_type comms_;
    std::uint64_t num_spikes_ = 0u;
};

} // namespace communication
} // namespace arb
