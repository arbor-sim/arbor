#include <queue>
#include <unordered_set>
#include <vector>

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

domain_decomposition partition_load_balance(
    const recipe& rec,
    const context& ctx,
    partition_hint_map hint_map)
{
    const bool gpu_avail = ctx->gpu->has_gpu();

    struct partition_gid_domain {
        partition_gid_domain(const gathered_vector<cell_gid_type>& divs, unsigned domains) {
            auto rank_part = util::partition_view(divs.partition());
            for (auto rank: count_along(rank_part)) {
                for (auto gid: util::subrange_view(divs.values(), rank_part[rank])) {
                    gid_map[gid] = rank;
                }
            }
        }

        int operator()(cell_gid_type gid) const {
            return gid_map.at(gid);
        }

        std::unordered_map<cell_gid_type, int> gid_map;
    };

    struct cell_identifier {
        cell_gid_type id;
        bool is_super_cell;
    };

    using util::make_span;

    unsigned num_domains = ctx->distributed->size();
    unsigned domain_id = ctx->distributed->id();
    auto num_global_cells = rec.num_cells();

    auto dom_size = [&](unsigned dom) -> cell_gid_type {
        const cell_gid_type B = num_global_cells/num_domains;
        const cell_gid_type R = num_global_cells - num_domains*B;
        return B + (dom<R);
    };

    // Global load balance

    std::vector<cell_gid_type> gid_divisions;
    auto gid_part = make_partition(
        gid_divisions, transform_view(make_span(num_domains), dom_size));

    // Local load balance

    std::vector<std::vector<cell_gid_type>> super_cells; //cells connected by gj
    std::vector<cell_gid_type> reg_cells; //independent cells

    // Map to track visited cells (cells that already belong to a group)
    std::unordered_set<cell_gid_type> visited;

    // Connected components algorithm using BFS
    std::queue<cell_gid_type> q;
    for (auto gid: make_span(gid_part[domain_id])) {
        if (!rec.gap_junctions_on(gid).empty()) {
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
                    auto conns = rec.gap_junctions_on(element);
                    for (auto c: conns) {
                        if (element != c.local.gid && element != c.peer.gid) {
                            throw bad_cell_description(cell_kind::cable, element);
                        }
                        cell_member_type other = c.local.gid == element ? c.peer : c.local;

                        if (!visited.count(other.gid)) {
                            visited.insert(other.gid);
                            q.push(other.gid);
                        }
                    }
                }
                super_cells.push_back(cg);
            }
        }
        else {
            // If cell has no gap_junctions, put in separate group of independent cells
            reg_cells.push_back(gid);
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
            if (cell.is_super_cell == false) {
                group_elements.push_back(cell.id);
            } else {
                if (group_elements.size() + super_cells[cell.id].size() > group_size && !group_elements.empty()) {
                    groups.push_back({k, std::move(group_elements), backend});
                    group_elements.clear();
                }
                for (auto gid: super_cells[cell.id]) {
                    group_elements.push_back(gid);
                }
            }
            if (group_elements.size()>=group_size) {
                groups.push_back({k, std::move(group_elements), backend});
                group_elements.clear();
            }
        }
        if (!group_elements.empty()) {
            groups.push_back({k, std::move(group_elements), backend});
        }
    }

    cell_size_type num_local_cells = local_gids.size();

    // Exchange gid list with all other nodes
    // global all-to-all to gather a local copy of the global gid list on each node.
    auto global_gids = ctx->distributed->gather_gids(local_gids);

    domain_decomposition d;
    d.num_domains = num_domains;
    d.domain_id = domain_id;
    d.num_local_cells = num_local_cells;
    d.num_global_cells = num_global_cells;
    d.groups = std::move(groups);
    d.gid_domain = partition_gid_domain(global_gids, num_domains);

    return d;
}

} // namespace arb

