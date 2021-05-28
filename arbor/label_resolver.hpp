#pragma once

#include <unordered_map>
#include <vector>

#include <arbor/common_types.hpp>

namespace arb {

// Data required for {gid, label} to lid resolution.
// (gids, sizes) and (labels, ranges) are expected to have the same size.
// `sizes` is a partitioning vector for associating a gid, with a (label, range) pair.
struct cell_label_range {
    // The gids of the cells.
    std::vector<cell_gid_type> gids;

    // The number of labels associated with each gid.
    std::vector<cell_size_type> sizes;

    // The labels corresponding to the gids, sorted relative to the gids.
    std::vector<cell_tag_type> labels;

    // The lid_range corresponding to each label.
    std::vector<lid_range> ranges;

    cell_label_range() = default;

    cell_label_range(std::vector<cell_gid_type> gids,
                     std::vector<cell_size_type> sizes,
                     std::vector<cell_tag_type> lbls,
                     std::vector<lid_range> rngs);

    void append(cell_label_range other);
};

// Struct used for selecting an lid of a {cell, label} pair according to an lid_selection_policy
struct label_resolver {
    struct range_set {
        std::vector<lid_range> ranges;
        std::vector<unsigned> ranges_partition = {0};

        bool operator==(const arb::label_resolver::range_set& other) const {
            return (ranges == other.ranges) && (ranges_partition == other.ranges_partition);
        }
    };

    using label_resolution_map = std::unordered_map<cell_tag_type, std::pair<range_set, cell_lid_type>>;
    mutable std::unordered_map<cell_gid_type, label_resolution_map> mapper;

    label_resolver() = delete;
    explicit label_resolver(cell_label_range);

    // Returns a vector of lids of a {gid, label} pair according to a policy.
    // The vector contains as many elements as identically names labels on the cell.
    cell_lid_type get_lid(const cell_global_label_type&) const;

    // Reset the current lid_indices to 0.
    void reset();
};

} // namespace arb
