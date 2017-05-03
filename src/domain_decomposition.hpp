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
    struct group_range_type {
        cell_gid_type from;
        cell_gid_type to;
        cell_kind kind;
    };

    domain_decomposition(const recipe& rec, const group_rules& rules) {
        EXPECTS(rules.target_group_size>0);

        auto num_domains = communication::global_policy::size();
        auto domain_id = communication::global_policy::id();

        // partition the cells globally across the domains
        num_global_cells_ = rec.num_cells();
        cell_begin_ = (cell_gid_type)(num_global_cells_*(domain_id/(double)num_domains));
        cell_end_ = (cell_gid_type)(num_global_cells_*((domain_id+1)/(double)num_domains));

        // partition the local cells into cell groups
        if (num_local_cells()>0) {
            cell_gid_type group_size = 1;
            group_starts_.push_back(cell_begin_);
            auto group_kind = rec.get_cell_kind(cell_begin_);
            group_kinds_.push_back(group_kind);
            cell_gid_type gid = cell_begin_+1;
            while (gid<cell_end_) {
                auto kind = rec.get_cell_kind(gid);
                if (kind!=group_kind || group_size>=rules.target_group_size) {
                    group_starts_.push_back(gid);
                    group_kinds_.push_back(kind);
                    group_size = 0;
                }
                ++group_size;
                ++gid;
            }
            group_starts_.push_back(cell_end_);
        }
    }

    util::optional<cell_gid_type> local_group_from_gid(cell_gid_type i) {
        // check if gid is a local cell
        if (!is_local_gid(i)) {
            return util::nothing;
        }
        return gid_group_partition().index(i);
    }

    cell_gid_type cell_begin() const {
        return cell_begin_;
    }

    cell_gid_type cell_end() const {
        return cell_end_;
    }

    cell_gid_type num_global_cells() const {
        return num_global_cells_;
    }

    cell_gid_type num_local_cells() const {
        return cell_end()-cell_begin();
    }

    cell_gid_type num_local_groups() const {
        return group_kinds_.size();
    }

    group_range_type get_group(std::size_t i) const {
        return {group_starts_[i], group_starts_[i+1], group_kinds_[i]};
    }

    bool is_local_gid(cell_gid_type i) const {
        return i>=cell_begin_ && i<cell_end_;
    }

    auto gid_group_partition() -> decltype(util::partition_view(std::vector<cell_gid_type>())) const {
        return util::partition_view(group_starts_);
    }

private:

    cell_gid_type cell_begin_;
    cell_gid_type cell_end_;
    cell_gid_type num_global_cells_;
    std::vector<cell_gid_type> group_starts_;
    std::vector<cell_kind> group_kinds_;
};

} // namespace mc
} // namespace nest
