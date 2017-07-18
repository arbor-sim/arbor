#pragma once

#include <type_traits>
#include <vector>
#include <unordered_map>

#include <backends.hpp>
#include <common_types.hpp>
#include <communication/global_policy.hpp>
#include <hardware/node.hpp>
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
class group_description {
    const cell_kind kind_;
    const std::vector<cell_gid_type> gids_;
    const backend_policy backend_;

public:
    group_description(cell_kind k, std::vector<cell_gid_type> g, backend_policy b):
        kind_(k), gids_(std::move(g)), backend_(b)
    {}

    cell_kind kind() const {
        return kind_;
    }

    backend_policy backend() const {
        return backend_;
    }

    const std::vector<cell_gid_type>& gids() const {
        return gids_;
    }
};

class domain_decomposition {
    cell_gid_type first_cell_on_domain(int dom) const {
        return (cell_gid_type)(num_global_cells_*(dom/(double)num_domains_));
    }

public:
    domain_decomposition(const recipe& rec, hw::node nd):
        node_(nd)
    {
        using kind_type = std::underlying_type<cell_kind>::type;
        using util::make_span;

        num_domains_ = communication::global_policy::size();
        domain_id_ = communication::global_policy::id();
        num_global_cells_ = rec.num_cells();

        //
        // GLOBAL LOAD BALANCE  /////////////////////////
        //

        gid_part_.reserve(num_domains_+1);
        for (auto d: make_span(0, num_domains_+1)) {
            gid_part_.push_back(first_cell_on_domain(d));
        }

        // Partition the cells globally across the domains.
        auto b = first_cell_on_domain(domain_id_);
        auto e = first_cell_on_domain(domain_id_+1);

        //
        // LOCAL LOAD BALANCE   /////////////////////////
        //

        std::unordered_map<kind_type, std::vector<cell_gid_type>> kind_lists;
        for (auto gid: make_span(b, e)) {
            kind_lists[rec.get_cell_kind(gid)].push_back(gid);
        }

        // Create a flat vector of the cell kinds present on this node,
        // partitioned such that kinds for which GPU implementation are
        // listed before the others.
        std::vector<cell_kind> kinds;
        for (auto l: kind_lists) {
            kinds.push_back(cell_kind(l.first));
        }
        std::partition(kinds.begin(), kinds.end(), has_gpu_backend);

        for (auto k: kinds) {
            // put all cells into a single cell group on the gpu if possible
            if (node_.num_gpus && has_gpu_backend(k)) {
                groups_.push_back({k, std::move(kind_lists[k]), backend_policy::gpu});
            }
            // otherwise place into cell groups of size 1 on the cpu cores
            else {
                for (auto gid: kind_lists[k]) {
                    groups_.push_back({k, {gid}, backend_policy::multicore});
                }
            }
        }
    }

    int gid_domain(cell_gid_type gid) const {
        EXPECTS(gid<num_global_cells_);
        auto& x = gid_part_;
        return std::distance(x.begin(), std::upper_bound(x.begin(), x.end(), gid))-1;
    }

    /// Returns the total number of cells in the global model.
    cell_size_type num_global_cells() const {
        return num_global_cells_;
    }

    /// Returns the number of cells on the local domain.
    cell_size_type num_local_cells() const {
        return gid_part_[domain_id_+1] - gid_part_[domain_id_];
    }

    /// Returns the number of cell groups on the local domain.
    cell_size_type num_local_groups() const {
        return groups_.size();
    }

    /// Returns meta data for a local cell group.
    const group_description& get_group(cell_size_type i) const {
        EXPECTS(i<num_local_groups());
        return groups_[i];
    }

    /// Tests whether a gid is on the local domain.
    bool is_local_gid(cell_gid_type i) const {
        return i>=gid_part_[domain_id_] && i<gid_part_[domain_id_+1];
    }

private:
    int num_domains_;
    int domain_id_;
    hw::node node_;
    cell_size_type num_global_cells_;
    std::vector<cell_gid_type> gid_part_;
    std::vector<cell_kind> group_kinds_;
    std::vector<group_description> groups_;
};


// The model needs to perform efficient lookup of properties associated
// with cell gid. Currently the only property is the cell group of the
// gid. The properties are stored in a gid_property struct.
class gid_prop_map {
public:
    struct prop {
        prop(cell_gid_type gid, cell_gid_type group): gid(gid), local_group(group) {}
        cell_gid_type gid;
        cell_gid_type local_group;
        friend bool operator<(prop l, prop r) {return l.gid<r.gid;};
    };

    gid_prop_map(const domain_decomposition& d) {
        props_.reserve(d.num_local_cells());
        for (auto i: util::make_span(0, d.num_local_groups())) {
            auto& g = d.get_group(i);
            for (auto gid: g.gids()) {
                props_.push_back({gid, i});
            }
        }
        util::sort(props_);
    }

    // Perform O(log(num_local_cells)) lookup of properties for a gid.
    // Use optional to indicate whether the cell with gid is local.
    util::optional<prop> get(cell_gid_type gid) const {
        auto it = std::lower_bound(props_.begin(), props_.end(), prop(gid, 0));
        if (it==props_.end()) {
            return util::nothing;
        }
        return  *it;
    }

private:
    std::vector<prop> props_;
};

} // namespace mc
} // namespace nest
