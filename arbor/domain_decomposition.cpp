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

domain_decomposition::domain_decomposition(const recipe& rec,
                                           context ctx,
                                           std::vector<group_description> groups):
    num_global_cells_{rec.num_cells()},
    groups_(std::move(groups))
{
    const auto* dist = ctx->distributed.get();
    num_domains_ = dist->size();
    domain_id_ = dist->id();
    const bool has_gpu = ctx->gpu->has_gpu();

    // Collect and do a first check on the local gid set
    // * Are all GJ connected cells in the same group
    std::vector<cell_gid_type> local_gids;
    for (const auto& g: groups_) {
        // Check whether GPU is supported and bail if not
        // TODO: This would benefit from generalisation; ie
        // bool context::has_backend(backend_kind kind)
        // bool compatible(cell_kind ck, backend_kind bk)
        if (g.backend == backend_kind::gpu) {
            if(!has_gpu) throw invalid_backend(domain_id_);
            if (g.kind != cell_kind::cable) throw incompatible_backend(domain_id_, g.kind);
        }
        // Check GJ cliques.
        std::unordered_set<cell_gid_type> gid_set(g.gids.begin(), g.gids.end());
        for (const auto& gid: gid_set) {
            if (gid >= num_global_cells_) throw out_of_bounds(gid, num_global_cells_);
            for (const auto& gj: rec.gap_junctions_on(gid)) {
                if (!gid_set.count(gj.peer.gid)) throw invalid_gj_cell_group(gid, gj.peer.gid);
            }
            local_gids.push_back(gid);
        }
    }
    num_local_cells_ = local_gids.size();

    // MPI: Build global gid list incl their partition into domains.
    auto global_gids = dist->gather_gids(local_gids);

    // Sanity check of global gid list
    // * missing GIDs?
    // * too many GIDs?
    // * duplicate GIDs?
    // * skipped GIDa?
    auto global_gid_vals = global_gids.values();
    util::sort(global_gid_vals);
    for (unsigned i = 1; i < global_gid_vals.size(); ++i) {
        if (global_gid_vals[i] == global_gid_vals[i-1]) {
            throw duplicate_gid(global_gid_vals[i]);
        }
        if (global_gid_vals[i] > global_gid_vals[i-1] + 1) {
            throw skipped_gid(global_gid_vals[i], global_gid_vals[i-1]);
        }
    }

    // Build map of local gid -> domain id (aka MPI rank)
    auto rank_part = util::partition_view(global_gids.partition());
    for (auto rank: count_along(rank_part)) {
        for (auto gid: util::subrange_view(global_gids.values(), rank_part[rank])) {
            gid_map_[gid] = rank;
        }
    }
}

int domain_decomposition::gid_domain(cell_gid_type gid) const { return gid_map_.at(gid); }

int domain_decomposition::num_domains() const { return num_domains_; }

int domain_decomposition::domain_id() const { return domain_id_; }

cell_size_type domain_decomposition::num_local_cells() const { return num_local_cells_; }

cell_size_type domain_decomposition::num_global_cells() const { return num_global_cells_; }

cell_size_type domain_decomposition::num_groups() const { return groups_.size(); }

const std::vector<group_description>& domain_decomposition::groups() const { return groups_; }

const group_description& domain_decomposition::group(unsigned idx) const {
    arb_assert(idx<num_groups());
    return groups_[idx];
}

} // namespace arb

