#include <numeric>
#include <utility>
#include <vector>
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
#include "network_impl.hpp"
#include "profile/profiler_macro.hpp"
#include "threading/threading.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "communication/communicator.hpp"

namespace arb {

communicator::communicator(const recipe& rec, const domain_decomposition& dom_dec, context ctx):
    num_total_cells_{rec.num_cells()},
    num_local_cells_{dom_dec.num_local_cells()},
    num_local_groups_{dom_dec.num_groups()},
    num_domains_{(cell_size_type)ctx->distributed->size()},
    ctx_(std::move(ctx)) {}

constexpr inline
bool is_external(cell_gid_type c) {
    // index of the MSB of cell_gid_type in bits
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // If set, we are external
    return bool(c & msb);
}

constexpr inline
cell_member_type global_cell_of(const cell_remote_label_type& c) {
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // set the MSB
    return {c.rid | msb, c.index};
}

constexpr inline
cell_member_type global_cell_of(const cell_member_type& c) {
    constexpr auto msb = static_cast<cell_gid_type>(1 << (std::numeric_limits<cell_gid_type>::digits - 1));
    // set the MSB
    return {c.gid | msb, c.index};
}

inline
void reset_index(const domain_decomposition& dom_dec,
                 std::vector<cell_size_type>& divs,
                 util::partition_view_type<std::vector<cell_size_type>>& part) {
    divs.clear();
    part = util::make_partition(divs,
                                util::transform_view(dom_dec.groups(),
                                                     [](const group_description& g){
                                                         return g.gids.size();
                                                     }));
}

inline
void reset_partition(const std::vector<std::vector<connection>>& connss,
                     std::vector<cell_size_type>& part) {
    part.clear();
    part.push_back(0);
    for (const auto& conns: connss) part.push_back(part.back() + conns.size());
}

void make_remote_connections(const std::vector<cell_gid_type>& gids,
                             const recipe& rec,
                             const domain_decomposition& dom_dec,
                             resolver& target_resolver,
                             resolver& source_resolver,
                             communicator::connection_list& out) {
    PE(init:communicator:update:connections:remote);
    std::vector<connection> ext_connections;
    std::size_t n_ext = 0;
    target_resolver.clear();
    for (auto tgt_gid: gids) {
        const auto iod = dom_dec.index_on_domain(tgt_gid);
        source_resolver.clear();
        for (const auto& conn: rec.external_connections_on(tgt_gid)) {
            auto src = global_cell_of(conn.source);
            auto src_gid = conn.source.rid;
            if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            ext_connections.push_back({src, tgt_lid, conn.weight, conn.delay, iod});
            ++n_ext;
        }
    }
    PL();

    PE(init:communicator:update:sort:remote);
    util::sort(ext_connections);
    PL();

    PE(init:communicator:update:destructure:remote);
    out.clear();
    out.reserve(n_ext);
    out.make(ext_connections);
    PL();
}

void communicator::update_connections(const recipe& rec,
                                      const domain_decomposition& dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    // Record all the gids in a flat vector.
    PE(init:communicator:update:collect_gids);
    std::vector<cell_gid_type> gids;
    gids.reserve(num_local_cells_);
    for (const auto& g: dom_dec.groups()) util::append(gids, g.gids);
    PL();

    // Prepare resolvers
    auto target_resolver = resolver(&target_resolution_map);
    auto source_resolver = resolver(&source_resolution_map);

    // Build cell partition by group for passing events to cell groups
    PE(init:communicator:update:index);
    reset_index(dom_dec, index_divisions_, index_part_);
    PL();

    // Construct connection from external
    make_remote_connections(gids, rec, dom_dec, target_resolver, source_resolver, ext_connections_);

    // Construct connections from recipe callback
    // NOTE: It'd great to parallelize here, however, as we write to different
    //       src_domains *and* use the same resolvers, that's not feasible.
    //       The only way to speed this up w/ more HW is to use more MPI tasks,
    //       for now. The alternative is
    //       - make coarsegrained parallel chunks,
    //       - copy the resolvers into each
    //       - generate one resultant vector each
    //       - merge those serially
    //       The word coarsegrained is load-bearing, as many small task will result
    //       in many, many allocations.
    PE(init:communicator:update:connections:local);
    std::size_t n_con = 0;
    std::vector<std::vector<connection>> connections_by_src_domain(num_domains_);
    target_resolver.clear();
    for (const auto tgt_gid: gids) {
        auto iod = dom_dec.index_on_domain(tgt_gid);
        source_resolver.clear();
        for (const auto& conn: rec.connections_on(tgt_gid)) {
            auto src_gid = conn.source.gid;
            if(src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(tgt_gid, src_gid, num_total_cells_);
            auto src_dom = dom_dec.gid_domain(src_gid);
            auto src_lid = source_resolver.resolve(conn.source);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            connections_by_src_domain[src_dom].emplace_back(cell_member_type(src_gid, src_lid), tgt_lid, conn.weight, conn.delay, iod);
            ++n_con;
        }
    }
    PL();

    // Construct connections from high-level specification.
    PE(init:communicator:update:connections:generated);
    for (const auto& conn: generate_connections(rec, ctx_, dom_dec)) {
        auto src_gid = conn.source.gid;
        // NOTE: a bit awkward, as we don't have the tgt_gid.
        if (src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(-1, src_gid, num_total_cells_);
        auto src_dom = dom_dec.gid_domain(src_gid);
        connections_by_src_domain[src_dom].push_back(conn);
        ++n_con;
    }
    PL();

    // Sort the connections for each domain; num_domains_ independent sorts
    // parallelized trivially.
    PE(init:communicator:update:sort:local);
    threading::parallel_for::apply(0, num_domains_, ctx_->thread_pool.get(),
                                   [&](auto i) { util::sort(connections_by_src_domain[i]); });
    PL();

    PE(init:communicator:update:connections:partition);
    reset_partition(connections_by_src_domain, connection_part_);
    PL();

    PE(init:communicator:update:destructure:local);
    connections_.clear();
    connections_.reserve(n_con);
    connections_.make(connections_by_src_domain);
    PL();
}

std::pair<cell_size_type, cell_size_type> communicator::group_queue_range(cell_size_type i) {
    arb_assert(i<num_local_groups_);
    return index_part_[i];
}

time_type communicator::min_delay() {
    time_type res = std::numeric_limits<time_type>::max();
    res = std::accumulate(connections_.delays.begin(), connections_.delays.end(),
                          res,
                          [](auto&& acc, time_type del) { return std::min(acc, del); });
    res = std::accumulate(ext_connections_.delays.begin(), ext_connections_.delays.end(),
                          res,
                          [](auto&& acc, time_type del) { return std::min(acc, del); });
    res = ctx_->distributed->min(res);
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
    auto global_spikes = ctx_->distributed->gather_spikes(local_spikes);
    num_spikes_ += global_spikes.size();
    PL();

    // Get remote spikes
    PE(communication:exchange:gather:remote);
    if (remote_spike_filter_) {
        local_spikes.erase(std::remove_if(local_spikes.begin(),
                                          local_spikes.end(),
                                          [this] (const auto& s) { return !remote_spike_filter_(s); }));
    }
    auto remote_spikes = ctx_->distributed->remote_gather_spikes(local_spikes);
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
void communicator::remote_ctrl_send_continue(const epoch& e) { ctx_->distributed->remote_ctrl_send_continue(e); }
void communicator::remote_ctrl_send_done() { ctx_->distributed->remote_ctrl_send_done(); }

template<typename S>
void append_events_from_domain(const communicator::connection_list& cons, size_t cn, const size_t ce,
                               const S& spks,
                               std::vector<pse_vector>& queues) {
    auto sp = spks.begin(), se = spks.end();
    // We have a choice of whether to walk spikes or connections:
    // i.e., we can iterate over the spikes, and for each spike search
    // the for connections that have the same source; or alternatively
    // for each connection, we can search the list of spikes for spikes
    // with the same source.
    //
    // We iterate over whichever set is the smallest, which has
    // complexity of order max(S log(C), C log(S)), where S is the
    // number of spikes, and C is the number of connections.
    // Thus the whole algorithm has O(min(S, C) log max(S, C))
    while (sp < se && cn < ce) {
        if ((ce - cn) < size_t(se - sp)) {
            auto src = cons.srcs[cn];
            // identify range of spikes to enqueue.
            auto fst = sp;
            if (fst->source != src) {
                fst = std::lower_bound(sp, se,
                                       src,
                                       [](const auto& spk, const auto& src) { return spk.source < src; });
            }
            for (; cn < ce && cons.srcs[cn] == src; ++cn) {
                auto dst = cons.dests[cn];
                auto del = cons.delays[cn];
                auto wgt = cons.weights[cn];
                auto dom = cons.idx_on_domain[cn];
                auto& que = queues[dom];
                // Handle all connections with the same source
                // scan the range of spikes, once per connection
                for (sp = fst; sp < se && sp->source == src; ++sp) {
                    que.emplace_back(dst, sp->time + del, wgt);
                }
            }
            // once we leave here, sp will be at the end of the eglible range
            // and all connections with the same source will have been treated.
            // so, we can just leave sp at this end.
        }
        else { // less spikes than connections, so iterate spikes linearly and bsearch connections
            auto spk = sp;
            auto src = spk->source;
            // Here, `cn` is the index of the first connection whose source
            // is larger or equal to the spike's source. It may be `ce` if
            // all elements compare < to spk.source.
            auto fst = cn;
            if (cons.srcs[fst] != src) {
                fst = std::lower_bound(cons.srcs.begin() + cn,
                                       cons.srcs.begin() + ce,
                                       src,
                                       [](auto a, auto b) { return a < b; })
                    - cons.srcs.begin();
            }
            for (sp = spk; sp < se && sp->source == src; ++sp) {
                for (cn = fst; cn < ce && cons.srcs[cn] == src; ++cn) {
                    auto dst = cons.dests[cn];
                    auto del = cons.delays[cn];
                    auto wgt = cons.weights[cn];
                    auto dom = cons.idx_on_domain[cn];
                    auto& que = queues[dom];
                    // If we ever get multiple spikes from the same source, treat
                    // them all. This is mostly rare.
                    // NB: Reset the spike iterator as we walk the same sub-range
                    // for each connection with the same source.
                    que.emplace_back(dst, sp->time + del, wgt);
                }
            }
        }
    }
}

void communicator::make_event_queues(communicator::spikes& spikes,
                                     std::vector<pse_vector>& queues) {
    arb_assert(queues.size()==num_local_cells_);
    const auto& sp = spikes.from_local.partition();
    const auto& cp = connection_part_;
    for (auto dom: util::make_span(num_domains_)) {
        append_events_from_domain(connections_, cp[dom], cp[dom+1],
                                  util::subrange_view(spikes.from_local.values(), sp[dom], sp[dom+1]),
                                  queues);
    }
    num_local_events_ = util::sum_by(queues, [](const auto& q) {return q.size();}, num_local_events_);
    // Now that all local spikes have been processed; consume the remote events coming in.
    // - turn all gids into externals
    std::for_each(spikes.from_remote.begin(), spikes.from_remote.end(),
                  [](auto& s) { s.source = global_cell_of(s.source); });
    append_events_from_domain(ext_connections_, 0, ext_connections_.size(), spikes.from_remote, queues);
}

std::uint64_t communicator::num_spikes() const { return num_spikes_; }
void communicator::set_num_spikes(std::uint64_t n) { num_spikes_ = n; }
cell_size_type communicator::num_local_cells() const { return num_local_cells_; }
const communicator::connection_list& communicator::connections() const { return connections_; }

void communicator::reset() {
    num_spikes_ = 0;
    num_local_events_ = 0;
}

} // namespace arb

