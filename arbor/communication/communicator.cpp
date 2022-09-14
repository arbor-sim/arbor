#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike.hpp>
#include <include/arbor/arbexcept.hpp>

#include "communication/gathered_vector.hpp"
#include "connection.hpp"
#include "distributed_context.hpp"
#include "execution_context.hpp"
#include "profile/profiler_macro.hpp"
#include "threading/threading.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "communication/communicator.hpp"

namespace arb {

communicator::communicator(const recipe& rec,
                           const domain_decomposition& dom_dec,
                           const label_resolution_map& source_resolution_map,
                           const label_resolution_map& target_resolution_map,
                           execution_context& ctx):  num_total_cells_{rec.num_cells()},
                                                     num_local_cells_{dom_dec.num_local_cells()},
                                                     num_local_groups_{dom_dec.num_groups()},
                                                     num_domains_{(cell_size_type) ctx.distributed->size()},
                                                     distributed_{ctx.distributed},
                                                     thread_pool_{ctx.thread_pool} {
    update_connections(rec, dom_dec, source_resolution_map, target_resolution_map);
}

void communicator::update_connections(const connectivity& rec,
                                      const domain_decomposition& dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    // Forget all lingering information
    connections_.clear();
    connection_part_.clear();
    index_divisions_.clear();

    // Make a list of local gid with their group index and connections
    //   -> gid_infos
    // Count the number of local connections (i.e. connections terminating on this domain)
    //   -> n_cons: scalar
    // Calculate and store domain id of the presynaptic cell on each local connection
    //   -> src_domains: array with one entry for every local connection
    // Also the count of presynaptic sources from each domain
    //   -> src_counts: array with one entry for each domain

    // Record all the gid in a flat vector.
    // These are used to map from local index to gid in the parallel loop
    // that populates gid_infos.
    std::vector<cell_gid_type> gids;
    gids.reserve(num_local_cells_);
    for (auto g: dom_dec.groups()) {
        util::append(gids, g.gids);
    }

// For caching information about each cell
    struct gid_info {
        cell_gid_type gid;                               // global identifier of cell
        cell_size_type index_on_domain;                  // index of cell in this domain
        std::vector<cell_connection> conns;              // list of connections terminating at this cell
        std::vector<external_cell_connection> ext_conns; // list of connections terminating at this cell                                    //
    };

    std::vector<gid_info> gid_infos(num_local_cells_, gid_info{});
    threading::parallel_for::apply(0, gids.size(), thread_pool_.get(),
        [&](cell_size_type i) {
            auto gid = gids[i];
            gid_infos[i] = {gid, i, rec.connections_on(gid), rec.external_connections_on(gid)};
        });

    cell_local_size_type
        n_cons = util::sum_by(gid_infos, [](const auto& g){ return g.conns.size(); }),
        n_ext_cons = util::sum_by(gid_infos, [](const auto& g){ return g.ext_conns.size(); });
    std::vector<unsigned> src_domains;
    src_domains.reserve(n_cons);
    std::vector<cell_size_type> src_counts(num_domains_);

    for (const auto& cell: gid_infos) {
        for (auto c: cell.conns) {
            auto sgid = c.source.gid;
            if (sgid >= num_total_cells_) {
                throw arb::bad_connection_source_gid(cell.gid, sgid, num_total_cells_);
            }
            const auto src = dom_dec.gid_domain(sgid);
            src_domains.push_back(src);
            src_counts[src]++;
        }
        // TODO Do we want/need to do something here for external connections?
    }

    // Construct the connections.
    // The loop above gave the information required to construct in place
    // the connections as partitioned by the domain of their source gid.
    connections_.resize(n_cons);
    ext_connections_.resize(n_ext_cons);
    util::make_partition(connection_part_, src_counts);
    std::size_t
        pos = 0,
        ext_pos = 0;
    auto offsets = connection_part_; // Copy, as we use this as the list of current target indices to write into
    auto src_domain = src_domains.begin();
    auto target_resolver = resolver(&target_resolution_map);
    for (const auto& cell: gid_infos) {
        auto index = cell.index_on_domain;
        auto source_resolver = resolver(&source_resolution_map);
        for (const auto& c: cell.conns) {
            auto src_lid = source_resolver.resolve(c.source);
            auto tgt_lid = target_resolver.resolve({cell.gid, c.dest});
            auto offset  = offsets[*src_domain]++;
            ++src_domain;
            connections_[offset] = {{c.source.gid, src_lid}, tgt_lid, c.weight, c.delay, index};
        }
        for (const auto& c: cell.ext_conns) {
            const auto i = ext_pos++;
            auto tgt_lid = target_resolver.resolve({cell.gid, c.dest});
            ext_connections_[i] = {c.source, tgt_lid, c.weight, c.delay, cell.index_on_domain};
        }
    }

    // Build cell partition by group for passing events to cell groups
    index_part_ = util::make_partition(index_divisions_,
        util::transform_view(
            dom_dec.groups(),
            [](const group_description& g){return g.gids.size();}));

    // Sort the connections for each domain.
    // This is num_domains_ independent sorts, so it can be parallelized trivially.
    const auto& cp = connection_part_;
    threading::parallel_for::apply(0, num_domains_, thread_pool_.get(),
        [&](cell_size_type i) {
            util::sort(util::subrange_view(connections_, cp[i], cp[i+1]));
        });
}

std::pair<cell_size_type, cell_size_type> communicator::group_queue_range(cell_size_type i) {
    arb_assert(i<num_local_groups_);
    return index_part_[i];
}

time_type communicator::min_delay() {
    auto local_min = std::accumulate(connections_.begin(), connections_.end(),
                                     std::numeric_limits<time_type>::max(),
                                     [](auto&& acc, auto&& el) { return std::min(acc, time_type(el.delay)); });
    return distributed_->min(local_min);
}

gathered_vector<spike> communicator::exchange(std::vector<spike> local_spikes) {
    PE(communication:exchange:sort);
    // sort the spikes in ascending order of source gid
    util::sort_by(local_spikes, [](spike s){return s.source;});
    PL();

    PE(communication:exchange:gather);
    // global all-to-all to gather a local copy of the global spike list on each node.
    auto global_spikes = distributed_->gather_spikes(local_spikes);
    num_spikes_ += global_spikes.size();
    PL();

    return global_spikes;
}

void communicator::make_event_queues(
        const gathered_vector<spike>& global_spikes,
        std::vector<pse_vector>& queues) {
    arb_assert(queues.size()==num_local_cells_);
    // Predicate for partitioning
    struct spike_pred {
        bool operator()(const spike& spk, const cell_member_type& src) { return spk.source < src; }
        bool operator()(const cell_member_type& src, const spike& spk) { return src < spk.source; }
    };
    const auto& sp = global_spikes.partition();
    const auto& cp = connection_part_;
    for (auto dom: util::make_span(num_domains_)) {
        auto cons = util::subrange_view(connections_,           cp[dom], cp[dom+1]);
        auto spks = util::subrange_view(global_spikes.values(), sp[dom], sp[dom+1]);
        auto sp = spks.begin(), se = spks.end();
        auto cn = cons.begin(), ce = cons.end();
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
                    queues[cn->index_on_domain()].push_back(make_event(*cn, s));
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
                    queues[c.index_on_domain()].push_back(make_event(c, *sp));
                }
                cn = targets.first;
                ++sp;
            }
        }
    }
}

std::uint64_t communicator::num_spikes() const {
    return num_spikes_;
}

void communicator::set_num_spikes(std::uint64_t n) {
    num_spikes_ = n;
}

cell_size_type communicator::num_local_cells() const {
    return num_local_cells_;
}

const std::vector<connection>& communicator::connections() const {
    return connections_;
}

const std::vector<ext_connection>& communicator::external_connections() const {
    return ext_connections_;
}

void communicator::reset() {
    num_spikes_ = 0;
}

} // namespace arb

