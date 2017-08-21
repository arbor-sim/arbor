#pragma once

#include <type_traits>
#include <vector>
#include <unordered_map>

#include <backends.hpp>
#include <common_types.hpp>
#include <communication/global_policy.hpp>
#include <hardware/node_info.hpp>
#include <recipe.hpp>
#include <util/optional.hpp>
#include <util/partition.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {

inline bool has_gpu_backend(cell_kind k) {
    if (k==cell_kind::cable1d_neuron) {
        return true;
    }
    return false;
}

/// Utility type for meta data for a local cell group.
struct group_description {
    const cell_kind kind;
    const std::vector<cell_gid_type> gids;
    const backend_kind backend;

    group_description(cell_kind k, std::vector<cell_gid_type> g, backend_kind b):
        kind(k), gids(std::move(g)), backend(b)
    {}
};

struct domain_decomposition {
    domain_decomposition(int num_dom, int dom_id,
                         cell_size_type n_local, cell_size_type n_global,
                         std::vector<group_description> grps):
        num_domains(num_dom),
        domain_id(dom_id),
        num_local_cells(n_local),
        num_global_cells(n_global),
        groups(std::move(grps))
    {}

    /// Return the domain id of cell with gid
    int gid_domain(cell_gid_type gid) const {
        EXPECTS(gid<num_global_cells_);
        return gid_part_.index(gid);
    }

    /// Tests whether a gid is on the local domain.
    bool is_local_gid(cell_gid_type gid) const {
        return gid_domain(gid)==domain_id;
    }

    const int num_domains;
    const int domain_id;
    const cell_size_type num_local_cells;
    const cell_size_type num_global_cells;
    const std::vector<group_description> groups;
};

} // namespace mc
} // namespace nest
