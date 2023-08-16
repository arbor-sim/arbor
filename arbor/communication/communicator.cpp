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

// Build local(ie Arbor to Arbor) connection list
// Writes
// * connections     := [connection]
// * connection_part := [index into connections]
//   - such that all connections _from the nth source domain_ are located
//     between connections_part[n] and connections_part[n+1] in connections.
//   - source domains are the MPI ranks associated with the gid of the source
//     of a connection.
//   - as the spike buffer is sorted and partitioned by said source domain, we
//     can use this to quickly filter spike buffer for spikes relevant to us.
// * index_divisions_ and index_part_
//   - index_part_ is used to map a cell group index to a range of queue indices
//   - these indices identify the queue in simulation belonging to cell the nth cell group
//   - queue stores incoming events for a cell
//   - events are not identical to spikes, but constructed from them
//   - this indirection is needed as communicator/simulation is responsible for multiple
//     cell groups.
//   - index_part_ is a view onto index_divisions_. The latter is not directly used, but is
//     the backing data of the former. (Essentially index_part[n] = range(index_div[n], index_div[n+1]))
void update_local_connections(const connectivity& rec,
                              const domain_decomposition& dec,
                              const std::vector<cell_gid_type>& gids,
                              size_t num_total_cells,
                              size_t num_local_cells,
                              size_t num_domains,
                              // Outputs; written into communicator
                              std::vector<connection>& connections_,
                              std::vector<cell_size_type>& connection_part_,
                              std::vector<cell_size_type>& index_divisions_,
                              util::partition_view_type<std::vector<cell_size_type>>& index_part_,
                              task_system_handle thread_pool_,
                              // Mutable state for label resolution.
                              resolver& target_resolver,
                              resolver& source_resolver) {
    PE(init:communicator:update:local:gid_connections);
    // List all connections and partition them by their _target cell's index_
    std::vector<cell_connection> gid_connections;
    std::vector<size_t> part_connections;
    part_connections.reserve(num_local_cells);
    part_connections.push_back(0);
    // Map connection _index_ to the id of the source gid's domain.
    // eg:
    //  Our gids    [23, 42], indices [0, 1] and #domain 3
    //  Connections [ 42 <- 0, 42 <- 1, 23 <- 5, 42 <- 23, 23 <- 1]
    //  Domains     [[0, 1, 2, 3], [4, 5], [...], [23, 42]]
    // Thus we get
    //  Src Domains [ 0, 1, 3, 0]
    //  Src Counts  [ 2, 1, 0, 1]
    //  Partitition [ 0 2 5 ]
    std::vector<unsigned> src_domains;
    std::vector<cell_size_type> src_counts(num_domains);

    // Build the data structures above.
    for (const auto gid: gids) {
        const auto& conns = rec.connections_on(gid);
        for (const auto& conn: conns) {
            const auto src_gid = conn.source.gid;
            if(is_external(src_gid)) throw arb::source_gid_exceeds_limit(gid, src_gid);
            if (src_gid >= num_total_cells) throw arb::bad_connection_source_gid(gid, src_gid, num_total_cells);
            const auto src = dec.gid_domain(src_gid);
            src_domains.push_back(src);
            src_counts[src]++;
            gid_connections.emplace_back(conn);
        }
        part_connections.push_back(gid_connections.size());
    }

    // Construct partitioning of connections on src_domains, thus
    // continuing the above example:
    // connection_part_ [ 0 2 3 3 4]
    // mapping the ranges
    // [0-2, 2-3, 3-3, 3-4]
    // in the to-be-created connection array.
    util::make_partition(connection_part_, src_counts);
    PL();

    // Construct the connections. The loop above gave us the information needed
    // to do this in place.
    PE(init:communicator:update:local:connections);
    connections_.resize(gid_connections.size());
    // Copy, as we use this as the list of current target indices to write into
    struct offset_t {
        std::vector<cell_size_type>::iterator source;
        std::vector<cell_size_type> offsets;
        cell_size_type next() { return offsets[*source++]++; }
    };

    auto offsets = offset_t{src_domains.begin(), connection_part_};
    for (const auto index: util::make_span(num_local_cells)) {
        const auto tgt_gid = gids[index];
        for (const auto cidx: util::make_span(part_connections[index],
                                              part_connections[index+1])) {
            const auto& conn = gid_connections[cidx];
            auto src_gid = conn.source.gid;
            auto src_lid = source_resolver.resolve(conn.source);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            auto out_idx = offsets.next();
            connections_[out_idx] = {{src_gid, src_lid}, tgt_lid, conn.weight, conn.delay, (cell_size_type)index};
        }
        source_resolver.reset();
    }
    PL();

    PE(init:communicator:update:local:index);
    // Build cell partition by group for passing events to cell groups
    index_part_ = util::make_partition(index_divisions_,
                                       util::transform_view(dec.groups(),
                                                            [](const auto& g){ return g.gids.size(); }));
    PL();

    PE(init:communicator:update:local:sort_connections);
    // Sort the connections for each domain.
    // This is num_domains_ independent sorts, so it can be parallelized trivially.
    threading::parallel_for::apply(0, num_domains, thread_pool_.get(),
                                   [&](cell_size_type i) {
                                       util::sort(util::subrange_view(connections_,
                                                                      connection_part_[i],
                                                                      connection_part_[i+1]));
                                   });
    PL();
}

// Build lists for the _remote_ connections. No fancy acceleration structures
// are built and the list is globally sorted.
void update_remote_connections(const connectivity& rec,
                               const domain_decomposition& dec,
                               const std::vector<cell_gid_type>& gids,
                               size_t num_total_cells,
                               size_t num_local_cells,
                               size_t num_domains,
                               // Outputs; written into communicator
                               std::vector<connection>& ext_connections_,
                               // Mutable state for label resolution.
                               resolver& target_resolver,
                               resolver& source_resolver) {
    PE(init:communicator:update:remote:gid_connections);
    std::vector<ext_cell_connection> gid_ext_connections;
    std::vector<size_t> part_ext_connections;
    part_ext_connections.reserve(num_local_cells);
    part_ext_connections.push_back(0);
    for (const auto gid: gids) {
        const auto& ext_conns = rec.external_connections_on(gid);
        for (const auto& conn: ext_conns) {
            // NOTE: This might look like a bug, but the _remote id_ is consider locally
            // in the remote id space, ie must not be already tagged as remote.
            if(is_external(conn.source.rid)) throw arb::source_gid_exceeds_limit(gid, conn.source.rid);
            gid_ext_connections.emplace_back(conn);
        }
        part_ext_connections.push_back(gid_ext_connections.size());
    }
    PL();

    // Construct the connections. The loop above gave us the information needed
    // to do this in place.
    PE(init:communicator:update:remote:connections);
    ext_connections_.resize(gid_ext_connections.size());
    std::size_t ext = 0;
    for (const auto index: util::make_span(num_local_cells)) {
        const auto tgt_gid = gids[index];
        for (const auto cidx: util::make_span(part_ext_connections[index],
                                              part_ext_connections[index+1])) {
            const auto& conn = gid_ext_connections[cidx];
            auto src = global_cell_of(conn.source);
            auto tgt_lid = target_resolver.resolve(tgt_gid, conn.target);
            ext_connections_[ext] = {src, tgt_lid, conn.weight, conn.delay, (cell_size_type) index};
            ++ext;
        }
        source_resolver.reset();
    }
    PL();

    PE(init:communicator:update:remote:sort_connections);
    std::sort(ext_connections_.begin(), ext_connections_.end());
    PL();
}

void communicator::update_connections(const connectivity& rec,
                                      const domain_decomposition& dom_dec,
                                      const label_resolution_map& source_resolution_map,
                                      const label_resolution_map& target_resolution_map) {
    PE(init:communicator:update:clear);
    // Forget all lingering information
    connections_.clear();
    connection_part_.clear();
    ext_connections_.clear();
    index_divisions_.clear();
    PL();

    // Remember list of local gids
    PE(init:communicator:update:collect_gids);
    std::vector<cell_gid_type> gids; gids.reserve(num_local_cells_);
    for (const auto& g: dom_dec.groups()) util::append(gids, g.gids);
    PL();

    // Build resolvers
    auto target_resolver = resolver(&target_resolution_map);
    auto source_resolver = resolver(&source_resolution_map);

    update_local_connections(rec, dom_dec, gids,
                             num_total_cells_, num_local_cells_, num_domains_,
                             connections_,
                             connection_part_, index_divisions_,
                             index_part_,
                             thread_pool_,
                             target_resolver, source_resolver);

    update_remote_connections(rec, dom_dec, gids,
                              num_total_cells_, num_local_cells_, num_domains_,
                              ext_connections_,
                              target_resolver, source_resolver);
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

