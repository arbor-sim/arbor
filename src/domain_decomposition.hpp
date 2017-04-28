#pragma once

#include <vector>

#include <common_types.hpp>
#include <communication/global_policy.hpp>
#include <recipe.hpp>
#include <util/optional.hpp>
#include <util/partition.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {

//TODO: interface for global gid_partition
//required by the communicator for one

class domain_decomposition {
public:
    using gid_type = cell_gid_type;

    struct group_range_type {
        gid_type from;
        gid_type to;
        cell_kind kind;
    };

    domain_decomposition(const recipe& rec, gid_type target_group_size) {
        EXPECTS(target_group_size>0);

        auto num_domains = communication::global_policy::size();
        auto domain_id = communication::global_policy::id();

        // partition the cells across the domain
        num_global_cells_ = rec.num_cells();
        first_cell_ = (gid_type)(num_global_cells_*(domain_id/(double)num_domains));
        last_cell_ = (gid_type)(num_global_cells_*((domain_id+1)/(double)num_domains));

        if (num_local_cells()>0) {
            gid_type group_size = 1;
            gid_type group_first = first_cell_;
            auto group_kind = rec.get_cell_kind(group_first);
            gid_type gid = group_first+1;
            while (gid<last_cell_) {
                auto kind = rec.get_cell_kind(gid);
                if (kind!=group_kind || group_size>=target_group_size) {
                    groups_.push_back({group_first, gid, group_kind});
                    group_kind = kind;
                    group_first = gid;
                    group_size = 0;
                }
                ++group_size;
                ++gid;
            }
            groups_.push_back({group_first, gid, group_kind});

            EXPECTS(groups_.front().from==first_cell_);
            EXPECTS(groups_.back().to==last_cell_);
        }
    }

    util::optional<gid_type> local_group_from_gid(gid_type i) {
        // check if gid is a local cell
        if (!is_local_gid(i)) {
            return util::nothing;
        }
        // TODO: finish binary search
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
        return groups_.size();
    }

    group_range_type get_group(gid_type i) const {
        return groups_[i];
    }

    bool is_local_gid(gid_type i) const {
        return i>=first_cell_ && i<last_cell_;
    }

    /*
    auto cell_partition() const {
        auto x =
            util::partition_view(
                util::transform_view(groups_, [](group_range_type g){return g.from;}));
    }
    */

private:

    gid_type first_cell_;
    gid_type last_cell_;
    gid_type num_global_cells_;
    std::vector<group_range_type> groups_;
};

} // namespace mc
} // namespace nest
