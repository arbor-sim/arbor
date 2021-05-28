#pragma once

#include <unordered_map>
#include <vector>

#include <arbor/common_types.hpp>

namespace arb {

// Data required for {gid, label} to lid resolution.
// `sizes` is a partitioning vector for associating a cell
// with a set of (label, range) pairs in `labels`, `ranges`.
class cell_label_range {
public:
    cell_label_range() = default;
    cell_label_range(cell_label_range&&) = default;
    cell_label_range(const cell_label_range&) = default;
    cell_label_range& operator=(const cell_label_range&) = default;

    cell_label_range(std::vector<cell_size_type> size_vec, std::vector<cell_tag_type> label_vec, std::vector<lid_range> range_vec);

    void add_cell();

    void add_label(cell_tag_type label, lid_range range);

    void append(cell_label_range other);

    bool check_invariant() const;

    const auto& sizes() const { return sizes_; }
    const auto& labels() const { return labels_; }
    const auto& ranges() const { return ranges_; }


private:
    // The number of labels associated with each cell.
    std::vector<cell_size_type> sizes_;

    // The labels corresponding to each cell, partitioned according to sizes_.
    std::vector<cell_tag_type> labels_;

    // The lid_range corresponding to each label.
    std::vector<lid_range> ranges_;
};

struct cell_labels_and_gids {
    cell_labels_and_gids() = default;
    cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gids);

    void append(cell_labels_and_gids other);

    bool check_invariant() const;

    cell_label_range label_range;
    std::vector<cell_gid_type> gids;
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
    explicit label_resolver(cell_labels_and_gids);

    // Returns a vector of lids of a {gid, label} pair according to a policy.
    // The vector contains as many elements as identically names labels on the cell.
    cell_lid_type get_lid(const cell_global_label_type&) const;

    // Reset the current lid_indices to 0.
    void reset();
};

} // namespace arb
