#pragma once

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Data required for {gid, label} to lid resolution.
// gids, labels, ranges are expected to have the same size.
// gids, labels, ranges are expected to be lexicographically sorted in that
// order within the boundaries of the partition vector. Typically, vectors
// are sorted per domain.
// For example if the partition vector is [0, 10, 20]:
// (gids, labels, ranges) are lexicographically sorted between (0, 10) and (10, 20).
struct cell_labeled_ranges {
    // The gids of the cells.
    std::vector<cell_gid_type> gids;

    // The number of labels associated with each gid.
    std::vector<cell_size_type> sizes;

    // The labels corresponding to each gid, sorted relative to the gids.
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

// Struct selecting an lid of a {cell, label} pair according to an lid_selection_policy
struct label_resolver {
    using label_resolution_map = std::unordered_map<cell_tag_type, std::pair<lid_range, cell_lid_type>>;
    mutable std::unordered_map<cell_gid_type, label_resolution_map> mapper;

    label_resolver() = delete;
    explicit label_resolver(cell_labeled_ranges);

    // Returns the lid of a {gid, label} pair according to a policy.
    // The optional rank index is used to accelerate the search for {gid, label}.
    cell_lid_type get_lid(const cell_global_label_type&) const;

    // Reset the indices vector to 0;
    void reset();
};

} // namespace arb
