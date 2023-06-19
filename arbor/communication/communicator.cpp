#include <numeric>
#include <utility>
#include <vector>
#include <any>
#include <limits>

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
#include "spike_source_cell_group.hpp"
#include "cable_cell_group.hpp"
#include "benchmark_cell_group.hpp"
#include "lif_cell_group.hpp"

#include "communication/communicator.hpp"

namespace arb {

communicator::communicator(const recipe& rec,
                           const domain_decomposition& dom_dec,
                           execution_context& ctx):  num_total_cells_{rec.num_cells()},
                                                     num_local_cells_{dom_dec.num_local_cells()},
                                                     num_local_groups_{dom_dec.num_groups()},
                                                     num_domains_{(cell_size_type) ctx.distributed->size()},
                                                     distributed_{ctx.distributed},
                                                     thread_pool_{ctx.thread_pool} {}

// This is a bit nasty, as we basically reimplement things from all cell kinds ...
cell_label_range get_sources(cell_gid_type gid, const recipe& rec) {
    auto cell = rec.get_cell_description(gid);
    auto kind = rec.get_cell_kind(gid);
    cell_label_range result;
    if (kind == cell_kind::lif) {
        get_sources(result, util::any_cast<lif_cell>(cell));
    }
    else if (kind == cell_kind::spike_source) {
        get_sources(result, util::any_cast<spike_source_cell>(cell));
    }
    else if (kind == cell_kind::benchmark) {
        get_sources(result, util::any_cast<benchmark_cell>(cell));
    }
    else if (kind == cell_kind::cable) {
        get_sources(result, util::any_cast<cable_cell>(cell));
    }
    else {
        throw arbor_internal_error("Unknown cell kind");
    }
    return result;
}    


constexpr inline
bool is_external(cell_gid_type c) {
    // index of the MSB of cell_gid_type in bits
    constexpr auto msb = 1 << (std::numeric_limits<cell_gid_type>::digits - 1);
    // If set, we are external
    return bool(c & msb);
}

constexpr inline
cell_member_type global_cell_of(const cell_remote_label_type& c) {
    constexpr auto msb = 1 << (std::numeric_limits<cell_gid_type>::digits - 1);
    // set the MSB
    return {c.rid | msb, c.index};
}

constexpr inline
cell_member_type global_cell_of(const cell_member_type& c) {
    constexpr auto msb = 1 << (std::numeric_limits<cell_gid_type>::digits - 1);
    // set the MSB
    return {c.gid | msb, c.index};
}

void communicator::update_connections(const recipe& rec,
                                      const domain_decomposition& dom_dec,
                                      const label_resolution_map& target_resolution_map) {
    PE(init:communicator:update:clear);
    // Forget all lingering information
    connections_.clear();
    connection_part_.clear();
    index_divisions_.clear();
    PL();

    // Make a list of local cells' connections
    //   -> gid_connections
    // Count the number of local connections (i.e. connections terminating on this domain)
    //   -> n_cons: scalar
    // Calculate and store domain id of the presynaptic cell on each local connection
    //   -> src_domains: array with one entry for every local connection
    // Also the count of presynaptic sources from each domain
    //   -> src_counts: array with one entry for each domain

    // Record all the gids in a flat vector.

    // Build the connection information for local cells.
    PE(init:communicator:update:gid_connections);
    std::vector<cell_connection> gid_connections;
    std::vector<ext_cell_connection> gid_ext_connections;
    std::vector<size_t> part_connections;
    part_connections.reserve(num_local_cells_);
    part_connections.push_back(0);
    std::vector<size_t> part_ext_connections;
    part_ext_connections.reserve(num_local_cells_);
    part_ext_connections.push_back(0);
    std::vector<unsigned> src_domains;
    std::vector<cell_size_type> src_counts(num_domains_);
    std::unordered_map<cell_gid_type, std::size_t> used;
    for (const auto& group: dom_dec.groups()) {
        for (const auto gid: group.gids) {
            // Local
            const auto& conns = rec.connections_on(gid);
            for (const auto& conn: conns) {
                const auto sgid = conn.source.gid;
                used[sgid] += 1;
                if (sgid >= num_total_cells_) throw arb::bad_connection_source_gid(gid, sgid, num_total_cells_);
                const auto src = dom_dec.gid_domain(sgid);
                src_domains.push_back(src);
                src_counts[src]++;
                gid_connections.emplace_back(conn);
            }
            part_connections.push_back(gid_connections.size());
            // Remote
            const auto& ext_conns = rec.external_connections_on(gid);
            for (const auto& conn: ext_conns) {
                gid_ext_connections.emplace_back(conn);
            }
            part_ext_connections.push_back(gid_ext_connections.size());
        }
    }

    // Construct a label resolver for a given gid.
    //
    // We could take care not to fetch sources from connected cells for those
    // gids that are local to or group. But, that doesn't seem to be worth the
    // extra effort.
    struct label_map {
        std::size_t used = 0;
        std::unique_ptr<label_resolution_map> map;
        resolver res;

        label_map(cell_gid_type gid, const recipe& rec):
            map{std::make_unique<label_resolution_map>(cell_labels_and_gids{get_sources(gid, rec), {gid}})},
            res{map.get()}
                {}
        cell_lid_type resolve(const cell_global_label_type& lbl) { used += 1; return res.resolve(lbl); }
        void reset() { res.reset(); }
        void clear() { res.clear(); map->clear(); }
    };

    // Construct the connections.
    // The loop above gave the information required to construct in place
    // the connections as partitioned by the domain of their source gid.
    util::make_partition(connection_part_, src_counts);
    auto n_cons = gid_connections.size();
    auto n_ext_cons = gid_ext_connections.size();
    PL();

    // Construct the connections. The loop above gave us the information needed
    // to do this in place.
    // NOTE: The connections are partitioned by the domain of their source gid.
    PE(init:communicator:update:connections);
    connections_.resize(n_cons);
    ext_connections_.resize(n_ext_cons);
    auto offsets = connection_part_; // Copy, as we use this as the list of current target indices to write into
    std::size_t ext = 0;
    auto src_domain = src_domains.begin();
    auto target_resolver = resolver(&target_resolution_map);
    auto sources = std::unordered_map<cell_gid_type, label_map>{};
    cell_size_type index = 0;
    for (const auto& group: dom_dec.groups()) {
        for (const auto& tgt_gid: group.gids) {
            for (const auto cidx: util::make_span(part_connections[index], part_connections[index+1])) {
                const auto& conn = gid_connections[cidx];
                auto src_gid = conn.source.gid;
                if (!sources.count(src_gid)) sources.emplace(src_gid, label_map{src_gid, rec});
                if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
                auto src_lid = sources.at(src_gid).resolve(conn.source);
                auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
                auto offset  = offsets[*src_domain]++;
                ++src_domain;
                connections_[offset] = {{src_gid, src_lid}, tgt_lid, conn.weight, conn.delay, index};
            }
            for (const auto cidx: util::make_span(part_ext_connections[index], part_ext_connections[index+1])) {
                const auto& conn = gid_ext_connections[cidx];
                auto src = global_cell_of(conn.source);
                auto src_gid = conn.source.rid;
                if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
                auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
                ext_connections_[ext] = {src, tgt_lid, conn.weight, conn.delay, index};
                ++ext;
            }
            for (auto& [k, v]: sources) {
                // To save a bit of peak memory, clear what we don't need anymore.
                if (v.used >= used[k]) {
                    v.clear();
                }
                else {
                    v.reset();
                }
            }
            ++index;
        }
    }
    PL();

    PE(init:communicator:update:index);
    // Build cell partition by group for passing events to cell groups
    index_part_ = util::make_partition(index_divisions_,
        util::transform_view(
            dom_dec.groups(),
            [](const group_description& g){ return g.gids.size(); }));
    PL();

    PE(init:communicator:update:sort_connections);
    // Sort the connections for each domain.
    // This is num_domains_ independent sorts, so it can be parallelized trivially.
    const auto& cp = connection_part_;
    threading::parallel_for::apply(0, num_domains_, thread_pool_.get(),
                                   [&](cell_size_type i) {
                                       util::sort(util::subrange_view(connections_, cp[i], cp[i+1]));
                                   });
    std::sort(ext_connections_.begin(), ext_connections_.end());
    PL();
}

std::pair<cell_size_type, cell_size_type> communicator::group_queue_range(cell_size_type i) {
    arb_assert(i<num_local_groups_);
    return index_part_[i];
}

time_type communicator::min_delay() {
    time_type res = std::numeric_limits<time_type>::max();
    res = std::accumulate(connections_.begin(), connections_.end(),
                                res,
                                [](auto&& acc, auto&& el) { return std::min(acc, time_type(el.delay)); });
    res = std::accumulate(ext_connections_.begin(), ext_connections_.end(),
                                res,
                                [](auto&& acc, auto&& el) { return std::min(acc, time_type(el.delay)); });
    res = distributed_->min(res);
    return res;
}

communicator::spikes
communicator::exchange(std::vector<spike> local_spikes) {
    PE(communication:exchange:sort);
    // sort the spikes in ascending order of source gid
    util::sort_by(local_spikes, [](spike s){return s.source;});
    PL();

    PE(communication:exchange:gather);
    // global all-to-all to gather a local copy of the global spike list on each node.
    auto global_spikes = distributed_->gather_spikes(local_spikes);
    num_spikes_ += global_spikes.size();
    PL();

    // Get remote spikes
    PE(communication:exchange:gather:remote);
    if (remote_spike_filter_) {
        local_spikes.erase(std::remove_if(local_spikes.begin(),
                                          local_spikes.end(),
                                          [this] (const auto& s) { return !remote_spike_filter_(s); }));
    }
    auto remote_spikes = distributed_->remote_gather_spikes(local_spikes);
    PL();

    PE(communication:exchange:gather:remote:post_process);
    // set the remote bit on all incoming spikes
    std::for_each(remote_spikes.begin(), remote_spikes.end(),
                  [](spike& s) { s.source = global_cell_of(s.source); });
    // sort, since we cannot trust our peers
    std::sort(remote_spikes.begin(), remote_spikes.end());
    PL();
    return {global_spikes, remote_spikes};
}

void communicator::set_remote_spike_filter(const spike_predicate& p) { remote_spike_filter_ = p; }
void communicator::remote_ctrl_send_continue(const epoch& e) { distributed_->remote_ctrl_send_continue(e); }
void communicator::remote_ctrl_send_done() { distributed_->remote_ctrl_send_done(); }

// Internal helper to append to the event queues
template<typename S, typename C>
void append_events_from_domain(C cons,
                               S spks,
                               std::vector<pse_vector>& queues) {
    // Predicate for partitioning
    struct spike_pred {
        bool operator()(const spike& spk, const cell_member_type& src) { return spk.source < src; }
        bool operator()(const cell_member_type& src, const spike& spk) { return src < spk.source; }
    };

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
    if (cons.size() < spks.size()) {
        while (cn != ce && sp != se) {
            auto sources = std::equal_range(sp, se, cn->source, spike_pred());
            for (auto s: util::make_range(sources)) {
                queues[cn->index_on_domain].push_back(make_event(*cn, s));
            }
            sp = sources.first;
            ++cn;
        }
    }
    else {
        while (cn != ce && sp != se) {
            auto targets = std::equal_range(cn, ce, sp->source);
            for (auto c: util::make_range(targets)) {
                queues[c.index_on_domain].push_back(make_event(c, *sp));
            }
            cn = targets.first;
            ++sp;
        }
    }
}

void communicator::make_event_queues(
        const gathered_vector<spike>& global_spikes,
        std::vector<pse_vector>& queues,
        const std::vector<spike>& external_spikes) {
    arb_assert(queues.size()==num_local_cells_);
    const auto& sp = global_spikes.partition();
    const auto& cp = connection_part_;
    for (auto dom: util::make_span(num_domains_)) {
        append_events_from_domain(util::subrange_view(connections_,           cp[dom], cp[dom+1]),
                                  util::subrange_view(global_spikes.values(), sp[dom], sp[dom+1]),
                                  queues);
    }
    num_local_events_ = util::sum_by(queues, [](const auto& q) {return q.size();}, num_local_events_);
    // Now that all local spikes have been processed; consume the remote events coming in.
    // - turn all gids into externals
    auto spikes = external_spikes;
    std::for_each(spikes.begin(),
                  spikes.end(),
                  [](auto& s) { s.source = global_cell_of(s.source); });
    append_events_from_domain(ext_connections_, spikes, queues);
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

void communicator::reset() {
    num_spikes_ = 0;
    num_local_events_ = 0;
}

} // namespace arb

