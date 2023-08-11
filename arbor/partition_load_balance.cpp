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
}

ARB_ARBOR_API domain_decomposition partition_load_balance(const recipe& rec,
                                                          context ctx,
                                                          const partition_hint_map& hint_map) {
    using util::make_span;

    const auto global_gj_connection_table = build_global_gj_connection_table(rec);

    const auto& dist = ctx->distributed;
    unsigned num_domains = dist->size();
    unsigned domain_id = dist->id();
    const bool gpu_avail = ctx->gpu->has_gpu();
    auto num_global_cells = rec.num_cells();

    auto dom_size = [&](unsigned dom) -> cell_gid_type {
        const cell_gid_type B = num_global_cells/num_domains;
        const cell_gid_type R = num_global_cells - num_domains*B;
        return B + (dom < R);
    };

    // Global load balance
    std::vector<cell_gid_type> gid_divisions;
    auto gid_part = make_partition(gid_divisions, transform_view(make_span(num_domains), dom_size));


    // Local load balance

    std::vector<std::vector<cell_gid_type>> super_cells; // cells connected by gj
    std::vector<cell_gid_type> reg_cells; //independent cells

    // Map to track visited cells (cells that already belong to a group)
    std::unordered_set<cell_gid_type> visited;

    // Connected components algorithm using BFS
    std::queue<cell_gid_type> q;
    for (auto gid: make_span(gid_part[domain_id])) {
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
            // If cell has no gap_junctions, put in separate group of independent cells
            reg_cells.emplace_back(gid);
        }
    }

    // Sort super_cell groups and only keep those where the first element in the group belongs to domain
    super_cells.erase(std::remove_if(super_cells.begin(), super_cells.end(),
            [gid_part, domain_id](std::vector<cell_gid_type>& cg)
            {
                std::sort(cg.begin(), cg.end());
                return cg.front() < gid_part[domain_id].first;
            }), super_cells.end());

    // Collect local gids that belong to this rank, and sort gids into kind lists
    // kind_lists maps a cell_kind to a vector of either:
    // 1. gids of regular cells (in reg_cells)
    // 2. indices of supercells (in super_cells)

    struct cell_identifier {
        cell_gid_type id;
        bool is_super_cell;
    };
    std::vector<cell_gid_type> local_gids;
    std::unordered_map<cell_kind, std::vector<cell_identifier>> kind_lists;
    for (auto gid: reg_cells) {
        local_gids.push_back(gid);
        kind_lists[rec.get_cell_kind(gid)].push_back({gid, false});
    }

    for (unsigned i = 0; i < super_cells.size(); i++) {
        auto kind = rec.get_cell_kind(super_cells[i].front());
        for (auto gid: super_cells[i]) {
            if (rec.get_cell_kind(gid) != kind) {
                throw gj_kind_mismatch(gid, super_cells[i].front());
            }
            local_gids.push_back(gid);
        }
        kind_lists[kind].push_back({i, true});
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
    for (auto l: kind_lists) {
        kinds.push_back(cell_kind(l.first));
    }
    std::partition(kinds.begin(), kinds.end(), has_gpu_backend);

    std::vector<group_description> groups;
    for (auto k: kinds) {
        partition_hint hint;
        if (auto opt_hint = util::value_by_key(hint_map, k)) {
            hint = opt_hint.value();
            if(!hint.cpu_group_size) {
                throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested cpu_cell_group size of {}", k, hint.cpu_group_size));
            }
            if(hint.prefer_gpu && !hint.gpu_group_size) {
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
        for (auto cell: kind_lists[k]) {
            if (!cell.is_super_cell) {
                group_elements.push_back(cell.id);
            } else {
                if (group_elements.size() + super_cells[cell.id].size() > group_size && !group_elements.empty()) {
                    groups.emplace_back(k, std::move(group_elements), backend);
                    group_elements.clear();
                }
                for (auto gid: super_cells[cell.id]) {
                    group_elements.push_back(gid);
                }
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

