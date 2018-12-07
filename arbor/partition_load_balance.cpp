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

namespace arb {

domain_decomposition partition_load_balance(
    const recipe& rec,
    const context& ctx,
    partition_hint_map hint_map)
{
    const bool gpu_avail = ctx->gpu->has_gpu();

    struct partition_gid_domain {
        partition_gid_domain(gathered_vector<cell_gid_type> divs, unsigned domains):
            gid_divisions(std::move(divs)), num_domains(domains)
        {}

        int operator()(cell_gid_type gid) const {
            auto gid_part = gid_divisions.partition();
            for(unsigned i = 0; i < num_domains; i++) {
                auto rank_gids = util::subrange_view(gid_divisions.values(), gid_part[i], gid_part[i+1]);
                auto lower = std::lower_bound(rank_gids.begin(), rank_gids.end(), gid);
                if(lower != rank_gids.end())
                    return i;
            }

        }

        const gathered_vector<cell_gid_type> gid_divisions;
        unsigned num_domains;
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

    std::vector<std::vector<cell_gid_type>> comp_groups; //compulsory groups
    std::vector<cell_gid_type> ind_cells; //independent cells

    // Map to track visited cells (cells that already belong to a group)
    std::unordered_map<cell_gid_type, bool> visited;

    // Connected components algorithm using BFS
    std::queue<cell_gid_type> q;
    for(auto gid: make_span(gid_part[domain_id])) {
        // If cell has no gap_junctions, put in separate group of independent cells
        if(!rec.gap_junctions_on(gid).empty()) {
            // If cell hasn't been visited yet, must belong to new group
            // Perform BFS starting from that cell
            if (visited.find(gid) == visited.end()) {
                std::vector<cell_gid_type> cg;
                q.push(gid);
                visited[gid] = true;
                while (!q.empty()) {
                    auto element = q.front();
                    q.pop();
                    cg.push_back(element);
                    // Adjacency list
                    auto conns = rec.gap_junctions_on(element);
                    for (auto c: conns) {
                        if (visited.find(c.location.gid) == visited.end()) {
                            q.push(c.location.gid);
                            visited[c.location.gid] = true;
                        }
                    }
                }
                comp_groups.push_back(cg);
            }
        }
        else {
            ind_cells.push_back(gid);
        }
    }

    // Sort groups and only keep those where the first element in the group belongs to domain
    comp_groups.erase(std::remove_if(comp_groups.begin(), comp_groups.end(),
            [gid_part, domain_id](std::vector<cell_gid_type>& cg)
            {
                std::sort(cg.begin(), cg.end());
                return cg.front() < gid_part[domain_id].first;
            }), comp_groups.end());

    // Collect local gids that belong to us, and sort independent gid into kind lists
    std::vector<cell_gid_type> local_gids;
    std::unordered_map<cell_kind, std::vector<cell_gid_type>> kind_lists;
    for (auto gid: ind_cells) {
        local_gids.push_back(gid);
        kind_lists[rec.get_cell_kind(gid)].push_back(gid);
    }

    for (auto cg: comp_groups) {
        for (auto gid: cg) {
            local_gids.push_back(gid);
        }
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
        }

        backend_kind backend = backend_kind::multicore;
        std::size_t group_size = hint.cpu_group_size;

        if (hint.prefer_gpu && gpu_avail && has_gpu_backend(k)) {
            backend = backend_kind::gpu;
            group_size = hint.gpu_group_size;
        }

        std::vector<cell_gid_type> group_elements;
        for (auto gid: kind_lists[k]) {
            group_elements.push_back(gid);
            if (group_elements.size()>=group_size) {
                groups.push_back({k, std::move(group_elements), backend});
                group_elements.clear();
            }
        }
        if (!group_elements.empty()) {
            groups.push_back({k, std::move(group_elements), backend});
        }
    }
    // comp_groups handled separately from ind_cells
    for (auto cg: comp_groups) {
        cell_kind k = rec.get_cell_kind(cg[0]);
        partition_hint hint;
        if (auto opt_hint = util::value_by_key(hint_map, k)) {
            hint = opt_hint.value();
        }
        backend_kind backend = backend_kind::multicore;
        if (hint.prefer_gpu && gpu_avail && has_gpu_backend(k)) {
            backend = backend_kind::gpu;
        }
        groups.push_back({k, std::move(cg), backend});
    }

    // Calculate the number of local cells
    cell_size_type num_local_cells = 0;
    for(auto g : groups) {
        num_local_cells += g.gids.size();
    }

    // Exchange gid list with all other nodes
    util::sort(local_gids);

    // global all-to-all to gather a local copy of the global gid list on each node.
    auto global_gids = ctx->distributed->gather_gids(local_gids);

    domain_decomposition d;
    d.num_domains = num_domains;
    d.domain_id = domain_id;
    d.num_local_cells = num_local_cells;
    d.num_global_cells = num_global_cells;
    d.groups = std::move(groups);
    d.gid_domain = partition_gid_domain(std::move(global_gids), num_domains); // TODO

    return d;
}

} // namespace arb

