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
#include "profile/profiler_macro.hpp"
#include "threading/threading.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

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

void communicator::update_connections(const connectivity& rec,
                                      const domain_decomposition& dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    PE(init:communicator:update:clear);
    // Forget all lingering information
    connections_.clear();
    ext_connections_.clear();
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

    // Record all the gid in a flat vector.

    PE(init:communicator:update:collect_gids);
    std::vector<cell_gid_type> gids; gids.reserve(num_local_cells_);
    for (const auto& g: dom_dec.groups()) util::append(gids, g.gids);
    PL();

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
    for (const auto gid: gids) {
        // Local
        const auto& conns = rec.connections_on(gid);
        for (const auto& conn: conns) {
            const auto sgid = conn.source.gid;
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

    util::make_partition(connection_part_, src_counts);
    auto n_cons = gid_connections.size();
    auto n_ext_cons = gid_ext_connections.size();
    PL();

    // Construct the connections. The loop above gave us the information needed
    // to do this in place.
    // NOTE: The connections are partitioned by the domain of their source gid.
    PE(init:communicator:update:connections);
    std::vector<connection> connections(n_cons);
    std::vector<connection> ext_connections(n_ext_cons);
    auto offsets = connection_part_; // Copy, as we use this as the list of current target indices to write into
    std::size_t ext = 0;
    auto src_domain = src_domains.begin();
    auto target_resolver = resolver(&target_resolution_map);
    for (const auto index: util::make_span(num_local_cells_)) {
        const auto tgt_gid = gids[index];
        auto source_resolver = resolver(&source_resolution_map);
        for (const auto cidx: util::make_span(part_connections[index], part_connections[index+1])) {
            const auto& conn = gid_connections[cidx];
            auto src_gid = conn.source.gid;
            if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
            auto src_lid = source_resolver.resolve(conn.source);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            auto offset  = offsets[*src_domain]++;
            ++src_domain;
            connections[offset] = {{src_gid, src_lid}, tgt_lid, conn.weight, conn.delay, index};
        }
        for (const auto cidx: util::make_span(part_ext_connections[index], part_ext_connections[index+1])) {
            const auto& conn = gid_ext_connections[cidx];
            auto src = global_cell_of(conn.source);
            auto src_gid = conn.source.rid;
            if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(tgt_gid, src_gid);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            ext_connections[ext] = {src, tgt_lid, conn.weight, conn.delay, index};
            ++ext;
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
                                       util::sort(util::subrange_view(connections, cp[i], cp[i+1]));
                                   });
    std::sort(ext_connections.begin(), ext_connections.end());
    PL();

    PE(init:communicator:update:destructure_connections);
    connections_.make(connections);
    ext_connections_.make(ext_connections);
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

template<typename It, typename Out>
It enqueue_from_source(const communicator::connection_list& cons,
                         const size_t idx,
                         It spk,
                         const It end,
                         Out& out) {
    // const refs to connection.
    auto src = cons.srcs[idx];
    auto dst = cons.dests[idx];
    auto del = cons.delays[idx];
    auto wgt = cons.weights[idx];
    // mutable reference to queue
    auto dom = cons.idx_on_domain[idx];
    for (; spk != end && spk->source == src; ++spk) {
        out.emplace_back(std::make_tuple(dom, pse_vector::value_type{dst, spk->time + del, wgt}));
    }
    return spk;
}
    
// Internal helper to append to the event queues
template<typename S>
void append_events_from_domain(const communicator::connection_list& cons,
                               size_t cn, const size_t ce,
                               const S& spks,
                               std::vector<pse_vector>& queues) {
    auto sp = spks.begin(), se = spks.end();
    std::vector<std::tuple<cell_size_type, pse_vector::value_type>> tmp;
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
        for (; cn < ce; ++cn) {
            sp = std::lower_bound(sp, se,
                                  cons.srcs[cn],
                                  [](const auto& spk, const auto& src) { return spk.source < src; });
            sp = enqueue_from_source(cons, cn, sp, se, tmp);
            if (sp == se) break;
        }
    }
    else {
        while (sp != se) {
            auto src = sp->source;
            // Here, `cn` is the index of the first connection whose source
            // is larger or equal to the spike's source. It may be `ce` if
            // all elements compare < to spk.source.
            cn = std::lower_bound(cons.srcs.begin() + cn,
                                  cons.srcs.begin() + ce,
                                  src)
                - cons.srcs.begin();
            for (;  cn < ce && cons.srcs[cn] == src; ++cn) {
                // If we ever get multiple spikes from the same source, treat
                // them all. This is mostly rare.
                enqueue_from_source(cons, cn, sp, se, tmp);
            }
            // Skip all spikes from the same source.
            while(sp != se && sp->source == src) ++sp;
        }
    }
    // NOTE How about we make an intermdiary vector containing (idx_on_doamin,
    // event). We then partition by the queue index = idx_on_domain. NExt, we
    // create one task per queue/partition which will take care of resizing and
    // appending the events. Partition is sufficient as events get sorted by
    // time later on.
    std::partition(tmp.begin(), tmp.end(), [](const auto& tp) { return tp.first(); });
    for (const auto& [dom, evt]: tmp) {
        queues[dom].push_back(evt);
    }
}

void communicator::make_event_queues(const gathered_vector<spike>& global_spikes,
                                     std::vector<pse_vector>& queues,
                                     const std::vector<spike>& external_spikes) {
    arb_assert(queues.size()==num_local_cells_);
    const auto& sp = global_spikes.partition();
    const auto& cp = connection_part_;
    for (auto dom: util::make_span(num_domains_)) {
        append_events_from_domain(connections_, cp[dom], cp[dom+1],
                                  util::subrange_view(global_spikes.values(), sp[dom], sp[dom+1]),
                                  queues);
    }
    num_local_events_ = util::sum_by(queues, [](const auto& q) {return q.size();}, num_local_events_);
    // Now that all local spikes have been processed; consume the remote events coming in.
    // - turn all gids into externals
    auto spikes = external_spikes;
    std::for_each(spikes.begin(), spikes.end(),
                  [](auto& s) { s.source = global_cell_of(s.source); });
    append_events_from_domain(ext_connections_, 0, ext_connections_.size(), spikes, queues);
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

