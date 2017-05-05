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
    cell_size_type target_group_size;
    backend_policy policy;
};

class domain_decomposition {
    using gid_partition_type =
        util::partition_range<std::vector<cell_gid_type>::const_iterator>;

public:
    /// Utility type for meta data for a local cell group.
    struct group_range_type {
        cell_gid_type begin;
        cell_gid_type end;
        cell_kind kind;
    };

    domain_decomposition(const recipe& rec, const group_rules& rules):
        backend_policy_(rules.policy)
    {
        EXPECTS(rules.target_group_size>0);

        auto num_domains = communication::global_policy::size();
        auto domain_id = communication::global_policy::id();

        // Partition the cells globally across the domains.
        num_global_cells_ = rec.num_cells();
        cell_begin_ = (cell_gid_type)(num_global_cells_*(domain_id/(double)num_domains));
        cell_end_ = (cell_gid_type)(num_global_cells_*((domain_id+1)/(double)num_domains));

        // Partition the local cells into cell groups that satisfy three
        // criteria:
        //  1. the cells in a group have contiguous gid
        //  2. the size of a cell group does not exceed rules.target_group_size;
        //  3. all cells in a cell group have the same cell_kind.
        // This simple greedy algorithm appends contiguous cells to a cell
        // group until either the target group size is reached, or a cell with a
        // different kind is encountered.
        // On completion, cell_starts_ partitions the local gid into cell
        // groups, and group_kinds_ records the cell kind in each cell group.
        if (num_local_cells()>0) {
            cell_size_type group_size = 1;

            // 1st group starts at cell_begin_
            group_starts_.push_back(cell_begin_);
            auto group_kind = rec.get_cell_kind(cell_begin_);

            // set kind for 1st group
            group_kinds_.push_back(group_kind);

            cell_gid_type gid = cell_begin_+1;
            while (gid<cell_end_) {
                auto kind = rec.get_cell_kind(gid);

                // Test if gid belongs to a new cell group, i.e. whether it has
                // a new cell_kind or if the target group size has been reached.
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

    /// Returns the local index of the cell_group that contains a cell with
    /// with gid.
    /// If the cell is not on the local domain, the optional return value is
    /// not set.
    util::optional<cell_size_type>
    local_group_from_gid(cell_gid_type gid) const {
        // check if gid is a local cell
        if (!is_local_gid(gid)) {
            return util::nothing;
        }
        return gid_group_partition().index(gid);
    }

    /// Returns the gid of the first cell on the local domain.
    cell_gid_type cell_begin() const {
        return cell_begin_;
    }

    /// Returns one past the gid of the last cell in the local domain.
    cell_gid_type cell_end() const {
        return cell_end_;
    }

    /// Returns the total number of cells in the global model.
    cell_size_type num_global_cells() const {
        return num_global_cells_;
    }

    /// Returns the number of cells on the local domain.
    cell_size_type num_local_cells() const {
        return cell_end()-cell_begin();
    }

    /// Returns the number of cell groups on the local domain.
    cell_size_type num_local_groups() const {
        return group_kinds_.size();
    }

    /// Returns meta data for a local cell group.
    group_range_type get_group(cell_size_type i) const {
        return {group_starts_[i], group_starts_[i+1], group_kinds_[i]};
    }

    /// Tests whether a gid is on the local domain.
    bool is_local_gid(cell_gid_type i) const {
        return i>=cell_begin_ && i<cell_end_;
    }

    /// Return a partition of the cell gid over local cell groups.
    gid_partition_type gid_group_partition() const {
        return util::partition_view(group_starts_);
    }

    /// Returns the backend policy.
    backend_policy backend() const {
        return backend_policy_;
    }

private:

    backend_policy backend_policy_;
    cell_gid_type cell_begin_;
    cell_gid_type cell_end_;
    cell_size_type num_global_cells_;
    std::vector<cell_size_type> group_starts_;
    std::vector<cell_kind> group_kinds_;
};

} // namespace mc
} // namespace nest
