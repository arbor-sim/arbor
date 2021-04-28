#pragma once

#include <unordered_map>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Data required for {gid, label} to lid resolution.
// (gids, sizes) and (labels, ranges) are expected to have the same size.
// `sizes` is a partitioning vector for associating a gid, with a (label, range) pair.
struct cell_labeled_ranges {
    // The gids of the cells.
    std::vector<cell_gid_type> gids;

    // The number of labels associated with each gid.
    std::vector<cell_size_type> sizes;

    // The labels corresponding to the gids, sorted relative to the gids.
    std::vector<cell_tag_type> labels;

    // The lid_range corresponding to each label.
    std::vector<lid_range> ranges;

    cell_labeled_ranges() = default;

    cell_labeled_ranges(std::vector<cell_gid_type> gids,
                        std::vector<cell_size_type> sizes,
                        std::vector<cell_tag_type> lbls,
                        std::vector<lid_range> rngs);

    void append(cell_labeled_ranges other);
};

// Struct used for selecting an lid of a {cell, label} pair according to an lid_selection_policy
struct label_resolver {
    using label_resolution_map = std::unordered_map<cell_tag_type, std::pair<lid_range, cell_lid_type>>;
    mutable std::unordered_map<cell_gid_type, label_resolution_map> mapper;

    label_resolver() = delete;
    explicit label_resolver(cell_labeled_ranges);

    // Returns the lid of a {gid, label} pair according to a policy.
    cell_lid_type get_lid(const cell_global_label_type&) const;

    // Reset the current lid_indices to 0.
    void reset();
};

} // namespace arb
