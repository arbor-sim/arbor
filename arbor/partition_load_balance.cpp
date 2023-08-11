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

namespace arb {

namespace {
// Build global GJ connectivity table such that
// * table[gid] is the set of all gids connected to gid via a GJ
// * iff A in table[B], then B in table[A]
auto build_global_gj_connection_table(const recipe& rec) {
    std::unordered_map<cell_gid_type, std::unordered_set<cell_gid_type>> res;
    for (cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) {
        for (const auto& gj: rec.gap_junctions_on(gid)) {
            res[gid].insert(gj.peer.gid);
        }
    }

    // Make all gj_connections bidirectional.
    for (auto& [gid, local_conns]: res) {
        for (auto peer: local_conns) {
            auto& peer_conns = res[peer];
            if (!peer_conns.count(gid)) peer_conns.insert(gid);
        }
    }
    return res;
}

auto make_local_gid_range(context ctx, cell_gid_type num_global_cells) {
    const auto& dist = ctx->distributed;
    unsigned num_domains = dist->size();
    unsigned domain_id = dist->id();

    auto dom_size = [&](unsigned dom) -> cell_gid_type {
        const cell_gid_type B = num_global_cells/num_domains;
        const cell_gid_type R = num_global_cells - num_domains*B;
        return B + (dom < R);
    };

    std::vector<cell_gid_type> gid_divisions;
    auto gid_part = util::make_partition(gid_divisions,
                                         util::transform_view(util::make_span(num_domains), dom_size));

    return gid_part[domain_id];
}

auto build_components(const std::unordered_map<cell_gid_type, std::unordered_set<cell_gid_type>>& global_gj_connection_table,
                      std::pair<cell_gid_type, cell_gid_type> local_gid_range) {
    std::vector<std::vector<cell_gid_type>> super_cells; // cells connected by gj
    std::vector<std::vector<cell_gid_type>> res;

    // track visited cells (cells that already belong to a group)
    std::unordered_set<cell_gid_type> visited;

    // Connected components algorithm using BFS
    std::queue<cell_gid_type> q;
    for (auto gid: util::make_span(local_gid_range)) {
        if (global_gj_connection_table.count(gid)) {
            // If cell hasn't been visited yet, must belong to new super_cell
            // Perform BFS starting from that cell
            if (!visited.count(gid)) {
                visited.insert(gid);
                std::vector<cell_gid_type> cg;
                q.push(gid);
                while (!q.empty()) {
                    auto element = q.front();
                    q.pop();
                    cg.push_back(element);
                    // Adjacency list
                    for (const auto& peer: global_gj_connection_table.at(element)) {
                        if (visited.insert(peer).second) q.push(peer);
                    }
                }
                super_cells.emplace_back(std::move(cg));
            }
        }
        else {
            // If cell has no gap_junctions, put in front
            res.push_back({gid});
        }
    }

    // Sort super_cell groups and only keep those where the first element in the group belongs to domain
    for (auto& sc: super_cells) {
        std::sort(sc.begin(), sc.end());
        if (!sc.empty() && sc.front() >= local_gid_range.first) res.push_back(sc);
    }
    return res;
}
}

ARB_ARBOR_API domain_decomposition partition_load_balance(const recipe& rec,
                                                          context ctx,
                                                          const partition_hint_map& hint_map) {
    const auto global_gj_connection_table = build_global_gj_connection_table(rec);
    const auto local_gid_range = make_local_gid_range(ctx, rec.num_cells());
    const auto components = build_components(global_gj_connection_table, local_gid_range);

    std::vector<cell_gid_type> local_gids;
    std::unordered_map<cell_kind, std::vector<cell_gid_type>> kind_lists;

    for (auto idx: util::make_span(components.size())) {
        const auto& component = components[idx];
        const auto& first_gid  = component.front();
        auto kind = rec.get_cell_kind(first_gid);
        for (auto gid: component) {
            if (rec.get_cell_kind(gid) != kind) throw gj_kind_mismatch(gid, first_gid);
            local_gids.push_back(gid);
        }
        kind_lists[kind].push_back((cell_gid_type) idx);
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

    auto has_gpu_backend = [&ctx](cell_kind c) { return ctx->gpu->has_gpu() && cell_kind_supported(c, backend_kind::gpu, *ctx); };

    std::vector<cell_kind> kinds;
    for (auto l: kind_lists) {
        kinds.push_back(l.first);
    }
    std::partition(kinds.begin(), kinds.end(), has_gpu_backend);

    std::vector<group_description> groups;
    for (auto k: kinds) {
        partition_hint hint;
        if (auto opt_hint = util::value_by_key(hint_map, k)) {
            hint = opt_hint.value();
            if(!hint.cpu_group_size) {
                throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested cpu_cell_group size of {}",
                                                         k, hint.cpu_group_size));
            }
            if(hint.prefer_gpu && !hint.gpu_group_size) {
                throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested gpu_cell_group size of {}",
                                                         k, hint.gpu_group_size));
            }
        }

        backend_kind backend = backend_kind::multicore;
        std::size_t group_size = hint.cpu_group_size;

        if (hint.prefer_gpu && has_gpu_backend(k)) {
            backend = backend_kind::gpu;
            group_size = hint.gpu_group_size;
        }

        std::vector<cell_gid_type> group_elements;
        // group_elements are sorted such that the gids of all members of a component are consecutive.
        for (auto cell: kind_lists[k]) {
            const auto& component = components[cell];
            if (group_elements.size() + component.size() > group_size && !group_elements.empty()) {
                groups.emplace_back(k, std::move(group_elements), backend);
                group_elements.clear();
            }
            for (auto gid: component) {
                group_elements.push_back(gid);
            }
            if (group_elements.size()>=group_size) {
                groups.emplace_back(k, std::move(group_elements), backend);
                group_elements.clear();
            }
        }
        if (!group_elements.empty()) {
            groups.emplace_back(k, std::move(group_elements), backend);
        }
    }

    return {rec, ctx, groups};
}
} // namespace arb

