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

namespace {
using gj_connection_set   = std::unordered_set<cell_gid_type>;
using gj_connection_table = std::unordered_map<cell_gid_type, gj_connection_set>;
using gid_range           = std::pair<cell_gid_type, cell_gid_type>;
using super_cell          = std::vector<cell_gid_type>;

// Build global GJ connectivity table such that
// * table[gid] is the set of all gids connected to gid via a GJ
// * iff A in table[B], then B in table[A]
auto build_global_gj_connection_table(const recipe& rec) {
    gj_connection_table res;
    // Collect all explicit GJ connections and make them bi-directional
    for (cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) {
        for (const auto& gj: rec.gap_junctions_on(gid)) {
            auto peer = gj.peer.gid;
            res[gid].insert(peer);
            res[peer].insert(gid);
        }
    }
    return res;
}

// compute range of gids for the local domain, such that the first (= num_cells
// % num_dom) domains get an extra element.
auto make_local_gid_range(context ctx, cell_gid_type num_global_cells) {
    const auto& dist = ctx->distributed;
    unsigned num_domains = dist->size();
    unsigned domain_id = dist->id();
    // normal block size
    auto block = num_global_cells/num_domains;
    // domains that need an extra element
    auto extra = num_global_cells - num_domains*block;
    // now compute the range
    if (domain_id < extra) {
        // all previous domains, incl ours, have an extra element
        auto beg = domain_id*(block + 1);
        auto end = beg + block + 1;
        return std::make_pair(beg, end);
    }
    else {
        // in this case the first `extra` domains added an extra element and the
        // rest has size `block`
        auto beg = extra + domain_id*block;
        auto end = beg + block;
        return std::make_pair(beg, end);
    }
}

// build the list of components for the local domain, where a component is a list of
// cell gids such that
// * the smallest gid in the list is in the local_gid_range
// * all gids that are connected to the smallest gid are also in the list
// * all gids w/o GJ connections come first (for historical reasons!?)
auto build_components(const gj_connection_table& global_gj_connection_table,
                      gid_range local_gid_range) {
    // cells connected by gj
    std::vector<super_cell> super_cells;
    // singular cells
    std::vector<super_cell> res;
    // track visited cells (cells that already belong to a group)
    gj_connection_set visited;
    // Connected components via BFS
    std::queue<cell_gid_type> q;
    for (auto gid: util::make_span(local_gid_range)) {
        if (global_gj_connection_table.count(gid)) {
            // If cell hasn't been visited yet, must belong to new component
            if (visited.insert(gid).second) {
                // pivot gid: the smallest found in this group; must be at
                // smaller or equal to `gid`.
                auto min_gid = gid;
                q.push(gid);
                super_cell sc;
                while (!q.empty()) {
                    auto element = q.front();
                    q.pop();
                    sc.push_back(element);
                    min_gid = std::min(element, min_gid);
                    // queue up conjoined cells
                    for (const auto& peer: global_gj_connection_table.at(element)) {
                        if (visited.insert(peer).second) q.push(peer);
                    }
                }
                // if the pivot gid belongs to our domain, this group will be part
                // of our domain, keep it and sort.
                if (min_gid >= local_gid_range.first) {
                    std::sort(sc.begin(), sc.end());
                    super_cells.emplace_back(std::move(sc));
                }
            }
        }
        else {
            res.push_back({gid});
        }
    }
    // append super cells to result
    res.insert(res.end(), super_cells.begin(), super_cells.end());
    return res;
}

// Figure what backend and group size to use
auto get_backend(context ctx, cell_kind kind, const partition_hint_map& hint_map) {
    auto has_gpu = ctx->gpu->has_gpu() && cell_kind_supported(kind, backend_kind::gpu, *ctx);
    const auto& hint = util::value_by_key_or(hint_map, kind, {});
    if (!hint.cpu_group_size) {
        throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested cpu_cell_group size of {}",
                                                 kind, hint.cpu_group_size));
    }
    if (hint.prefer_gpu && !hint.gpu_group_size) {
        throw arbor_exception(arb::util::pprintf("unable to perform load balancing because {} has invalid suggested gpu_cell_group size of {}",
                                                 kind, hint.gpu_group_size));
    }
    if (hint.prefer_gpu && has_gpu) return std::make_pair(backend_kind::gpu, hint.gpu_group_size);
    return std::make_pair(backend_kind::multicore, hint.cpu_group_size);
}

struct group_parameters {
    cell_kind kind;
    backend_kind backend;
    size_t size;
};

// Create a flat vector of the cell kinds present on this node, sorted such that
// kinds for which GPU implementation are listed before the others. This is a
// very primitive attempt at scheduling; the cell_groups that run on the GPU
// will be executed before other cell_groups, which is likely to be more
// efficient.
//
// TODO: This creates an dependency between the load balancer and the threading
// internals. We need support for setting the priority of cell group updates
// according to rules such as the back end on which the cell group is running.
auto build_group_parameters(context ctx,
                            const partition_hint_map& hint_map,
                            const std::unordered_map<cell_kind, std::vector<cell_gid_type>>& kind_lists) {
    std::vector<group_parameters> res;
    for (const auto& [kind, _gids]: kind_lists) {
        const auto& [backend, group_size] = get_backend(ctx, kind, hint_map);
        res.push_back({kind, backend, group_size});
    }
    util::sort_by(res, [](const auto& p) { return p.kind; });
    return res;
}

// Build the list of GJ-connected cells local to this domain.
// NOTE We put this into its own function to avoid increasing RSS.
auto build_local_components(const recipe& rec, context ctx) {
    const auto global_gj_connection_table = build_global_gj_connection_table(rec);
    const auto local_gid_range = make_local_gid_range(ctx, rec.num_cells());
    return build_components(global_gj_connection_table, local_gid_range);
}

} // namespace

ARB_ARBOR_API domain_decomposition partition_load_balance(const recipe& rec,
                                                          context ctx,
                                                          const partition_hint_map& hint_map) {
    const auto components = build_local_components(rec, ctx);

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

    auto kinds = build_group_parameters(ctx, hint_map, kind_lists);

    std::vector<group_description> groups;
    for (const auto& params: kinds) {
        std::vector<cell_gid_type> group_elements;
        // group_elements are sorted such that the gids of all members of a component are consecutive.
        for (auto cell: kind_lists[params.kind]) {
            const auto& component = components[cell];
            // adding the current group would go beyond alloted size, so add to the list
            // of groups and start a new one.
            if (group_elements.size() + component.size() > params.size && !group_elements.empty()) {
                groups.emplace_back(params.kind, std::move(group_elements), params.backend);
                group_elements.clear();
            }
            // we are clear to add the current component. NOTE this may exceed
            // the alloted size, but only by the minimal amount manageable
            group_elements.insert(group_elements.end(), component.begin(), component.end());
        }
        // we may have a trailing, incomplete group, so add it.
        if (!group_elements.empty()) groups.emplace_back(params.kind, std::move(group_elements), params.backend);
    }

    return {rec, ctx, groups};
}
} // namespace arb

