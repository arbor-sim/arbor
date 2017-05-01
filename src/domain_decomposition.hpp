#pragma once

#include <vector>

#include <backends.hpp>
#include <common_types.hpp>
#include <communication/global_policy.hpp>
#include <recipe.hpp>
#include <util/optional.hpp>
#include <util/partition.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {

// Meta data used to guide the domain_decomposition in distributing
// and grouping cells.
struct group_rules {
    cell_gid_type target_group_size;
    backend_policy policy;
};

class domain_decomposition {
public:
    using gid_type = cell_gid_type;

    struct group_range_type {
        gid_type from;
        gid_type to;
        cell_kind kind;
    };

    domain_decomposition(const recipe& rec, const group_rules& rules) {
        EXPECTS(rules.target_group_size>0);

        auto num_domains = communication::global_policy::size();
        auto domain_id = communication::global_policy::id();

        // partition the cells globally across the domains
        num_global_cells_ = rec.num_cells();
        first_cell_ = (gid_type)(num_global_cells_*(domain_id/(double)num_domains));
        last_cell_ = (gid_type)(num_global_cells_*((domain_id+1)/(double)num_domains));

        // partition the local cells into cell groups
        if (num_local_cells()>0) {
            gid_type group_size = 1;
            group_starts_.push_back(first_cell_);
            auto group_kind = rec.get_cell_kind(first_cell_);
            group_kinds_.push_back(group_kind);
            gid_type gid = first_cell_+1;
            while (gid<last_cell_) {
                auto kind = rec.get_cell_kind(gid);
                if (kind!=group_kind || group_size>=rules.target_group_size) {
                    group_starts_.push_back(gid);
                    group_kinds_.push_back(kind);
                    group_size = 0;
                }
                ++group_size;
                ++gid;
            }
            group_starts_.push_back(last_cell_);
        }
    }

    util::optional<gid_type> local_group_from_gid(gid_type i) {
        // check if gid is a local cell
        if (!is_local_gid(i)) {
            return util::nothing;
        }
        return local_gid_partition().index(i);
    }

    gid_type first_cell() const {
        return first_cell_;
    }

    gid_type last_cell() const {
        return last_cell_;
    }

    gid_type num_global_cells() const {
        return last_cell_;
    }

    gid_type num_local_cells() const {
        // valid when the invariant last_cell >= first_cell is satisfied
        return last_cell()-first_cell();
    }

    gid_type num_local_groups() const {
        return group_kinds_.size();
    }

    group_range_type get_group(gid_type i) const {
        return {group_starts_[i], group_starts_[i+1], group_kinds_[i]};
    }

    bool is_local_gid(gid_type i) const {
        return i>=first_cell_ && i<last_cell_;
    }

    auto local_gid_partition() -> decltype(util::partition_view(std::vector<gid_type>())) const {
        return util::partition_view(group_starts_);
    }

private:

    gid_type first_cell_;
    gid_type last_cell_;
    gid_type num_global_cells_;
    std::vector<gid_type> group_starts_;
    std::vector<cell_kind> group_kinds_;
};

} // namespace mc
} // namespace nest
