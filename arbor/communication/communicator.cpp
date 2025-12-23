#include <numeric>
#include <utility>
#include <vector>
#include <limits>
#include <unordered_set>

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

communicator::communicator(const recipe& rec, const domain_decomposition_ptr dom_dec, context ctx):
    num_total_cells_{rec.num_cells()},
    num_local_cells_{dom_dec->num_local_cells()},
    num_local_groups_{dom_dec->num_groups()},
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
void reset_index(const domain_decomposition_ptr dom_dec,
                 std::vector<cell_size_type>& divs,
                 util::partition_view_type<std::vector<cell_size_type>>& part) {
    divs.clear();
    part = util::make_partition(divs,
                                util::transform_view(dom_dec->groups(),
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
                             const domain_decomposition_ptr dom_dec,
                             resolver& target_resolver,
                             resolver& source_resolver,
                             communicator::connection_list& out) {
    PE(connections);
    std::vector<connection> ext_connections;
    std::size_t n_ext = 0;
    target_resolver.clear();
    for (auto tgt_gid: gids) {
        const auto iod = dom_dec->index_on_domain(tgt_gid);
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
    PL(connections);

    PE(sort);
    util::sort(ext_connections);
    PL(sort);

    PE(destructure);
    out.clear();
    out.reserve(n_ext);
    out.make(ext_connections);
    PL(destructure);
}

void communicator::update_connections(const recipe& rec,
                                      const domain_decomposition_ptr dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    PE(update_conns);
    // Record all the gids in a flat vector.
    PE(collect_gids);
    std::vector<cell_gid_type> gids;
    gids.reserve(num_local_cells_);
    for (const auto& g: dom_dec->groups()) util::append(gids, g.gids);
    PL(collect_gids);

    // Prepare resolvers
    auto target_resolver = resolver(&target_resolution_map);
    auto source_resolver = resolver(&source_resolution_map);

    // Build cell partition by group for passing events to cell groups
    PE(index);
    reset_index(dom_dec, index_divisions_, index_part_);
    PL(index);

    // Construct connection from external
    PE(remote);
    make_remote_connections(gids, rec, dom_dec, target_resolver, source_resolver, ext_connections_);
    PL(remote);

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
    //       in many, many allocations and we don't have the proper primitives.
    std::size_t n_con = 0;
    std::vector<std::vector<connection>> connections_by_src_domain(num_domains_);
    std::vector<std::vector<cell_member_type>> gids_domains(num_domains_);
    for (auto& v : gids_domains) {
        v.reserve(gids.size());
    }
    // helper for adding a connection
    auto push_connection = [&] (const auto& conn, cell_gid_type tgt_gid, cell_size_type tgt_iod) {
        auto src_gid = conn.source.gid;
        if(src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(tgt_gid, src_gid, num_total_cells_);
            // strip off qualifiers and match on type to find the actual lid of source
            auto src_lid = cell_lid_type(-1);
            using C = std::decay_t<decltype(conn)>;
            if constexpr (std::is_same_v<cell_connection, C>) {
                src_lid = source_resolver.resolve(conn.source);
            }
            else if constexpr (std::is_same_v<raw_cell_connection, C>) {
                src_lid = conn.source.index;
            }
            else {
                ARB_UNREACHABLE;
            }
            // targets always get resolution
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            // NOTE old compilers stumble over emplace_back here
            auto src_dom = dom_dec->gid_domain(src_gid);
            connections_by_src_domain[src_dom].emplace_back(
                connection{
                .source={.gid=src_gid, .index=src_lid},
                .target=tgt_lid,
                .weight=conn.weight,
                .delay=conn.delay,
                .index_on_domain=tgt_iod
            });
            gids_domains[src_dom].push_back(cell_member_type{src_gid, src_lid});
    };

    PE(local);
    target_resolver.clear();
    bool resolution_enabled = rec.resolve_sources();
    for (const auto tgt_gid: gids) {
        auto tgt_iod = dom_dec->index_on_domain(tgt_gid);
        source_resolver.clear();
        for (const auto& conn: rec.connections_on(tgt_gid)) {
            if (!resolution_enabled) throw resolution_disabled{tgt_gid};
            push_connection(conn, tgt_gid, tgt_iod);
            ++n_con;
        }
    }
    PL(local);

    PE(raw);
    for (const auto tgt_gid: gids) {
        auto tgt_iod = dom_dec->index_on_domain(tgt_gid);
        for (const auto& conn: rec.raw_connections_on(tgt_gid)) {
            push_connection(conn, tgt_gid, tgt_iod);
            ++n_con;
        }
    }
    PL(raw);


    // Construct connections from high-level specification.
    PE(generated);
    for (const auto& conn: generate_connections(rec, ctx_, dom_dec)) {
        auto src_gid = conn.source.gid;
        // NOTE: a bit awkward, as we don't have the tgt_gid.
        if (src_gid >= num_total_cells_) throw arb::bad_connection_source_gid(-1, src_gid, num_total_cells_);
        auto src_dom = dom_dec->gid_domain(src_gid);
        connections_by_src_domain[src_dom].push_back(conn);
        gids_domains[src_dom].push_back(conn.source);
        ++n_con;
    }
    PL(generated);

    PE(sort_unique);
    arb::threading::parallel_for::apply(0, gids_domains.size(), ctx_->thread_pool.get(),
                                        [&](int i) {
                                          auto& domain_gids = gids_domains[i];
                                          std::sort(domain_gids.begin(), domain_gids.end());
                                          domain_gids.erase(
                                              std::unique(domain_gids.begin(), domain_gids.end()),
                                              domain_gids.end()
                                          );
                                        });
    PL(sort_unique);

    PE(gids);
    auto srcs_by_rank = ctx_->distributed->all_to_all_gids_domains(gids_domains);
    const auto& part = srcs_by_rank.partition();
    const auto& srcs = srcs_by_rank.values();
    for (auto domain: util::make_span(0, num_domains_)) {
      auto beg = part[domain];
      auto end = part[domain + 1];
      for (auto idx: util::make_span(beg, end)) {
        const auto& src = srcs[idx];
        src_ranks_[src].push_back(domain);
      }
    }
    PL(gids);

    // Sort the connections for each domain; num_domains_ independent sorts
    // parallelized trivially.
    PE(sort_local);
    threading::parallel_for::apply(0, num_domains_, ctx_->thread_pool.get(),
                                   [&](auto i) { util::sort(connections_by_src_domain[i]); });
    PL(sort_local);

    PE(partition);
    reset_partition(connections_by_src_domain, connection_part_);
    PL(partition);

    PE(destructure);
    connections_.clear();
    connections_.reserve(n_con);
    connections_.make(connections_by_src_domain);
    PL(destructure);
    PL(update_conns);
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

gathered_vector<spike>
generate_all_to_all_vector(const std::vector<spike>& spikes,
                           const std::unordered_map<cell_member_type, std::vector<cell_size_type>>& src_ranks,
                           std::size_t num_domains) {

    using count_type = gathered_vector<spike>::count_type;
    // count outgoing spikes per rank
    std::vector<count_type> offsets(num_domains + 1, 0);
    for (const auto& spk: spikes) {
        auto ranks = src_ranks.find(spk.source);
        if (ranks != src_ranks.end()) {
            for (auto rank: ranks->second) {
                ++offsets[rank + 1];
            }
        }
    }

    // make partition so we can sort the spikes into bins
    std::partial_sum(offsets.begin(), offsets.end(),
                     offsets.begin());
    auto size = offsets.back();

    // we have the sizes per rank to send to, so deal spikes into bins.
    std::vector<spike> spikes_per_rank(size);
    auto rank_indices = offsets;
    for (const auto& spk: spikes) {
        auto ranks = src_ranks.find(spk.source);
        if (ranks != src_ranks.end()) {
            for (auto rank: ranks->second) {
                auto& index = rank_indices[rank];
                spikes_per_rank[index] = spk;
                ++index;
            }
        }
    }
    return {std::move(spikes_per_rank), std::move(offsets)};
}

communicator::spikes
communicator::exchange(std::vector<spike>& local_spikes) {
    PE(exchange);
    PE(sort);
    // sort the spikes in ascending order of source gid
    util::sort_by(local_spikes, [](spike s){return s.source;});
    PL(sort);

    PE(sum_spikes);
    num_local_spikes_ = ctx_->distributed->sum(local_spikes.size());
    num_spikes_ += num_local_spikes_;
    PL(sum_spikes);

    PE(generate);
    auto spikes_per_rank = generate_all_to_all_vector(local_spikes, src_ranks_, num_domains_);
    PL(generate);
    PE(all2all);
    // global all-to-all to gather a local copy of the global spike list on each node.
    auto global_spikes = ctx_->distributed->all_to_all_spikes(spikes_per_rank);
    PL(all2all);

    // Get remote spikes
    PE(remote);
    if (remote_spike_filter_) {
        local_spikes.erase(std::remove_if(local_spikes.begin(),
                                          local_spikes.end(),
                                          [this] (const auto& s) { return !remote_spike_filter_(s); }));
    }
    auto remote_spikes = ctx_->distributed->remote_gather_spikes(local_spikes);
    PL(remote);

    PE(post_process);
    // set the remote bit on all incoming spikes
    std::for_each(remote_spikes.begin(), remote_spikes.end(),
                  [](spike& s) { s.source = global_cell_of(s.source); });
    // sort, since we cannot trust our peers
    std::sort(remote_spikes.begin(), remote_spikes.end());
    PL(post_process);
    PL(exchange);
    return {std::move(global_spikes), std::move(remote_spikes)};
}

void communicator::set_remote_spike_filter(const spike_predicate& p) { remote_spike_filter_ = p; }
void communicator::remote_ctrl_send_continue(const epoch& e) { ctx_->distributed->remote_ctrl_send_continue(e); }
void communicator::remote_ctrl_send_done() { ctx_->distributed->remote_ctrl_send_done(); }

template<typename S>
void append_events_from_domain(const communicator::connection_list& cons, size_t cn, const size_t ce,
                               const S& spks,
                               std::vector<pse_vector>& queues) {
    auto sp = spks.begin(), se = spks.end();
    while (sp < se && cn < ce) {
        auto src = cons.srcs[cn];
        while (sp < se && sp->source < src) ++sp;
        if (sp >= se) continue;
        auto fst = sp;
        for (; cn < ce && cons.srcs[cn] == src; ++cn) {
            auto dom = cons.idx_on_domain[cn];
            auto& que = queues[dom];
            auto dst = cons.dests[cn];
            auto del = cons.delays[cn];
            auto wgt = cons.weights[cn];
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
std::uint64_t communicator::num_local_spikes() const { return num_local_spikes_; }
void communicator::set_num_spikes(std::uint64_t n) { num_spikes_ = n; }
cell_size_type communicator::num_local_cells() const { return num_local_cells_; }
const communicator::connection_list& communicator::connections() const { return connections_; }

void communicator::reset() {
    num_spikes_ = 0;
    num_local_events_ = 0;
}

} // namespace arb

