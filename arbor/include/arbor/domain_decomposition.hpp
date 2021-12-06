#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

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
};

/// Meta data that describes a domain decomposition.
/// A domain_decomposition type is responsible solely for describing the
/// distribution of cells across cell_groups and domains.
/// A load balancing algorithm is responsible for generating the
/// domain_decomposition, e.g. arb::partitioned_load_balancer().
struct domain_decomposition {
    /// Return the domain id of cell with gid.
    /// Supplied by the load balancing algorithm that generates the domain
    /// decomposition.
    std::function<int(cell_gid_type)> gid_domain;

    /// Number of distrubuted domains
    int num_domains;

    /// The index of the local domain
    int domain_id;

    /// Total number of cells in the local domain
    cell_size_type num_local_cells;

    /// Total number of cells in the global model (sum over all domains)
    cell_size_type num_global_cells;

    /// Descriptions of the cell groups on the local domain
    std::vector<group_description> groups;
};

} // namespace arb
