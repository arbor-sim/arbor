#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/domdecexcept.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/symmetric_recipe.hpp>
#include <arbor/context.hpp>

#include "cell_group_factory.hpp"
#include "execution_context.hpp"
#include "gpu_context.hpp"
#include "util/maputil.hpp"
#include "util/partition.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

#include <iostream>

namespace arb {

auto get_domain_range(std::size_t id, std::size_t global, std::size_t count) {
    // NOTE this seems like quite a bit of work for a simple thing
    auto size = [&](unsigned dom) -> cell_gid_type {
        const cell_gid_type B = global/count;
        const cell_gid_type R = global - count*B;
        return B + (dom<R);
    };

    std::vector<cell_gid_type> divisions;
    auto part = make_partition(divisions, transform_view(util::make_span(count), size));
    return part[id];
}

auto
build_clusters(std::size_t dom_beg, std::size_t dom_end,
               const std::vector<std::vector<cell_gid_type>>& global_gj_table) {
    // Cells connected by gj
    std::vector<std::vector<cell_gid_type>> super_cells;
    // Map to track visited cells (cells that already belong to a group)
    std::unordered_set<cell_gid_type> visited;

    // Connected components algorithm using BFS
    std::queue<cell_gid_type> q;
    for (auto gid: util::make_span(dom_beg, dom_end)) {
        if (visited.count(gid)) continue;
        // If cell hasn't been visited yet, must belong to new super_cell.
        // Perform BFS starting from that cell
        visited.insert(gid);
        std::vector<cell_gid_type> cg;
        q.push(gid);
        while (!q.empty()) {
            auto element = q.front();
            q.pop();
            cg.push_back(element);
            // Adjacency list
            for (const auto& peer: global_gj_table[element]) {
                if (visited.insert(peer).second) {
                    q.push(peer);
                }
            }
        }
        // Sort super_cell groups and only keep those where the first
        // element in the group belongs to domain
        util::sort(cg);
        if (!cg.empty() && cg.front() >= dom_beg) {
            super_cells.emplace_back(std::move(cg));
        }
    }
    return super_cells;
}

auto
make_gj_bidrectional(std::vector<std::vector<cell_gid_type>>& global_gj_table) {
    std::vector<std::unordered_set<cell_gid_type>> missing_peers(global_gj_table.size());
    for (auto gid: util::count_along(global_gj_table)) {
        const auto& local_conns = global_gj_table[gid];
        for (auto peer: local_conns) {
            auto& peer_conns = global_gj_table[peer];
            // If gid is not in the peer connection table insert it into the
            // missing_peers set
            if (!std::binary_search(peer_conns.begin(), peer_conns.end(), gid)) {
                missing_peers[peer].insert(gid);
            }
        }
    }
    // Append the missing peers into the global_gj_connections table
    for (auto ix: util::count_along(global_gj_table)) {
        std::move(missing_peers[ix].begin(), missing_peers[ix].end(),
                  std::back_inserter(global_gj_table[ix]));
    }
}

auto make_global_gj_table(const recipe& rec) {
    // The table is indexed by the index of the target gid in the gid_part of that domain.
    auto count = rec.num_cells();
    std::vector<std::vector<cell_gid_type>> local_gj_connection_table(count);
    for (auto gid: util::make_span(count)) {
        auto& target = local_gj_connection_table[gid];
        for (const auto& gj: rec.gap_junctions_on(gid)) target.push_back(gj.peer.gid);
        util::sort(target);
    }
    return local_gj_connection_table;
}

ARB_ARBOR_API domain_decomposition
partition_load_balance(const recipe& rec,
                       context ctx,
                       const partition_hint_map& hint_map) {
    const auto& dist = ctx->distributed;
    unsigned num_domains = dist->size();
    unsigned domain_id = dist->id();
    const bool gpu_avail = ctx->gpu->has_gpu();
    auto num_global_cells = rec.num_cells();

    const auto& [dom_beg, dom_end] = get_domain_range(domain_id, num_global_cells, num_domains);

    // Global load balance
    auto global_gj_connection_table = make_global_gj_table(rec);
    make_gj_bidrectional(global_gj_connection_table);

    const auto& super_cells = build_clusters(dom_beg, dom_end, global_gj_connection_table);

    // Local load balance

    // Collect local gids that belong to this rank, and sort gids into kind lists
    // kind_lists maps a cell_kind to a vector of gid
    std::vector<cell_gid_type> local_gids;
    std::unordered_map<cell_kind, std::vector<cell_gid_type>> kind_lists;
    for (auto ix: util::count_along(super_cells)) {
        auto& super_cell = super_cells[ix];
        auto kind = rec.get_cell_kind(super_cell.front());
        for (auto gid: super_cell) {
            if (rec.get_cell_kind(gid) != kind) {
                throw gj_kind_mismatch(gid, super_cell.front());
            }
            local_gids.push_back(gid);
        }
        kind_lists[kind].push_back(static_cast<cell_gid_type>(ix));
    }

    // Create a flat vector of the cell kinds present on this node,
    // partitioned such that kinds for which GPU implementation are
    // listed before the others. This is a very primitive attempt at
    // scheduling; the cell_groups that run on the GPU will be executed
    // before other cell_groups, which is likely to be more efficient.
    //
    // TODO: This creates an dependency between the load balancer and
    // the threading internals. We need support for setting the priority
    // of cell group updates according to rules such as the back end on
    // which the cell group is running.

    auto has_gpu_backend = [&ctx](cell_kind c) {
        return cell_kind_supported(c, backend_kind::gpu, *ctx);
    };

    std::vector<cell_kind> kinds;
    for (const auto& [k, v]: kind_lists) kinds.push_back(cell_kind(k));
    std::partition(kinds.begin(), kinds.end(), has_gpu_backend);

    std::vector<group_description> groups;
    for (auto k: kinds) {
        partition_hint hint;
        if (auto opt_hint = util::value_by_key(hint_map, k)) {
            hint = opt_hint.value();
            if (!hint.cpu_group_size) {
                throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested cpu_cell_group size of {}", k, hint.cpu_group_size));
            }
            if (hint.prefer_gpu && !hint.gpu_group_size) {
                throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested gpu_cell_group size of {}", k, hint.gpu_group_size));
            }
        }

        backend_kind backend = backend_kind::multicore;
        std::size_t group_size = hint.cpu_group_size;

        if (hint.prefer_gpu && gpu_avail && has_gpu_backend(k)) {
            backend = backend_kind::gpu;
            group_size = hint.gpu_group_size;
        }

        std::vector<cell_gid_type> group_elements;
        // group_elements are sorted such that the gids of all members of a super_cell are consecutive.
        for (auto& cell: kind_lists[k]) {
            auto& super_cell = super_cells[cell];
            group_elements.insert(group_elements.end(), super_cell.begin(), super_cell.end());
            if (group_elements.size()>=group_size) {
                groups.emplace_back(k, std::move(group_elements), backend);
                group_elements.clear();
            }
        }
        if (!group_elements.empty()) {
            groups.emplace_back(k, std::move(group_elements), backend);
        }
    }

    return domain_decomposition(rec, ctx, std::move(groups));
}

} // namespace arb

