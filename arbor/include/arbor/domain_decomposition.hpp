#pragma once

#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/export.hpp>
#include <arbor/recipe.hpp>
#include <arbor/serdes.hpp>

namespace arb {

/// Metadata for a local cell group.
struct group_description {
    /// The kind of cell in the group. All cells in a cell_group have the same type.
    cell_kind kind;

    /// The gids of the cells in the cell_group. Does not need to be sorted.
    std::vector<cell_gid_type> gids;

    /// The back end on which the cell_group is to run.
    backend_kind backend;

    group_description(cell_kind k, std::vector<cell_gid_type> g, backend_kind b):
        kind(k), gids(std::move(g)), backend(b)
    {}

    ARB_SERDES_ENABLE(group_description, kind, gids, backend);
};

/// Meta data that describes a domain decomposition.
/// A domain_decomposition type is responsible solely for describing the
/// distribution of cells across cell_groups and domains.
/// A load balancing algorithm is responsible for generating the
/// domain_decomposition, e.g. arb::partitioned_load_balancer().
struct ARB_ARBOR_API domain_decomposition {
    domain_decomposition() = delete;
    domain_decomposition(const recipe& rec, context ctx, const std::vector<group_description>& groups);

    domain_decomposition(const domain_decomposition&) = default;
    domain_decomposition& operator=(const domain_decomposition&) = default;

    int gid_domain(cell_gid_type gid) const { return gid_domain_[gid]; }
    cell_size_type index_on_domain(cell_gid_type gid) const { return gid_index_[gid]; }
    int num_domains() const;
    int domain_id() const;
    cell_size_type num_local_cells() const;
    cell_size_type num_global_cells() const;
    cell_size_type num_groups() const;
    const std::vector<group_description>& groups() const;
    const group_description& group(unsigned) const;

private:
    /// Return the domain id and index on domain of cell with gid.
    std::vector<int> gid_domain_;
    std::vector<cell_size_type> gid_index_;

    /// Number of distributed domains
    int num_domains_;

    /// The index of the local domain
    int domain_id_;

    /// Total number of cells in the local domain
    cell_size_type num_local_cells_;

    /// Total number of cells in the global model (sum over all domains)
    cell_size_type num_global_cells_;

    /// Descriptions of the cell groups on the local domain
    std::vector<group_description> groups_;
};

using domain_decomposition_ptr = std::shared_ptr<domain_decomposition>;

} // namespace arb
