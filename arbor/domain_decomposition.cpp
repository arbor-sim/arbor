#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/domdecexcept.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>
#include <arbor/context.hpp>

#include "execution_context.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {
domain_decomposition::domain_decomposition(
    const recipe& rec,
    context ctx,
    const std::vector<group_description>& groups)
{
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

    const auto* dist = ctx->distributed.get();
    unsigned num_domains = dist->size();
    int domain_id = dist->id();
    cell_size_type num_global_cells = rec.num_cells();
    const bool has_gpu = ctx->gpu->has_gpu();

    std::vector<cell_gid_type> local_gids;
    for (const auto& g: groups) {
        if (g.backend == backend_kind::gpu && !has_gpu) {
            throw invalid_backend(domain_id);
        }
        if (g.backend == backend_kind::gpu && g.kind != cell_kind::cable) {
            throw incompatible_backend(domain_id, g.kind);
        }

        std::unordered_set<cell_gid_type> gid_set(g.gids.begin(), g.gids.end());
        for (const auto& gid: g.gids) {
            if (gid >= num_global_cells) {
                throw out_of_bounds(gid, num_global_cells);
            }
            for (const auto& gj: rec.gap_junctions_on(gid)) {
                if (!gid_set.count(gj.peer.gid)) {
                    throw invalid_gj_cell_group(gid, gj.peer.gid);
                }
            }
        }
        local_gids.insert(local_gids.end(), g.gids.begin(), g.gids.end());
    }
    cell_size_type num_local_cells = local_gids.size();

    auto global_gids = dist->gather_gids(local_gids);
    if (global_gids.size() != num_global_cells) {
        throw invalid_sum_local_cells(global_gids.size(), num_global_cells);
    }

    auto global_gid_vals = global_gids.values();
    util::sort(global_gid_vals);
    for (unsigned i = 1; i < global_gid_vals.size(); ++i) {
        if (global_gid_vals[i] == global_gid_vals[i-1]) {
            throw duplicate_gid(global_gid_vals[i]);
        }
    }

    num_domains_ = num_domains;
    domain_id_ = domain_id;
    num_local_cells_ = num_local_cells;
    num_global_cells_ = num_global_cells;
    groups_ = groups;
    gid_domain_ = partition_gid_domain(global_gids, num_domains);
}

int domain_decomposition::gid_domain(cell_gid_type gid) const {
    return gid_domain_(gid);
}

int domain_decomposition::num_domains() const {
    return num_domains_;
}

int domain_decomposition::domain_id() const {
    return domain_id_;
}

cell_size_type domain_decomposition::num_local_cells() const {
    return num_local_cells_;
}

cell_size_type domain_decomposition::num_global_cells() const {
    return num_global_cells_;
}

cell_size_type domain_decomposition::num_groups() const {
    return groups_.size();
}

const std::vector<group_description>& domain_decomposition::groups() const {
    return groups_;
}

const group_description& domain_decomposition::group(unsigned idx) const {
    arb_assert(idx<num_groups());
    return groups_[idx];
}

} // namespace arb

