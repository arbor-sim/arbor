#pragma once

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Data required for {gid, label} to lid resolution.
// gids, labels, ranges are expected to have the same size.
// gids, labels, ranges are expected to be lexicographically sorted in the following order (gid, labels, ranges)
// within the boundaries of the partition vector. Typically, vectors are sorted per domain.
//
// For example if the partition vector is [0, 10, 20]:
// (gids, labels, ranges) are lexicographically sorted between (0, 10) and (10, 20).
struct cell_labeled_ranges {
    // The gids of the cells, with one entry per label on the cell.
    std::vector<cell_gid_type> gids;

    // The labels on the cells.
    std::vector<cell_tag_type> labels;

    // The range of possible indices corresponding to the {gid, label} pair
    std::vector<lid_range> ranges;

    // Partitioning of lexicographically sorted ranges. This is to avoid sorting across multiple ranks.
    std::vector<std::size_t> sorted_partitions;

    cell_labeled_ranges() = default;

    cell_labeled_ranges(std::vector<cell_gid_type> gids,
                       std::vector<cell_tag_type> lbls,
                       std::vector<lid_range> rngs,
                       std::vector<size_t> ptns);

    // Expects sorted tuple_vec
    explicit cell_labeled_ranges(const std::vector<std::tuple<cell_gid_type, std::string, lid_range>>& tuple_vec);

    bool is_one_partition() const;
    std::optional<std::pair<std::size_t, std::size_t>> get_gid_range(cell_gid_type, int partition=-1) const;
    std::optional<std::pair<std::size_t, std::size_t>> get_label_range(const cell_tag_type&, std::pair<std::size_t, std::size_t>) const;
};

enum class lid_selection_policy {
    round_robin,
    assert_univalent
};

// Struct selecting an lid of a {cell, label} pair according to an lid_selection_policy
struct label_resolver {
    cell_labeled_ranges mapper;
    mutable std::vector<cell_lid_type> indices;

    label_resolver() = delete;
    explicit label_resolver(cell_labeled_ranges);

    cell_lid_type get_lid(const cell_label_type&, lid_selection_policy=lid_selection_policy::round_robin) const;
    cell_lid_type get_lid(const cell_label_type&, int rank, lid_selection_policy=lid_selection_policy::round_robin) const;
};

} // namespace arb
