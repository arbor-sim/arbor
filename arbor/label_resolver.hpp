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
struct label_resolution_map {
    struct range_set {
        std::vector<lid_range> ranges;
        std::vector<unsigned> ranges_partition = {0};

        bool operator==(const label_resolution_map::range_set& other) const {
            return (ranges == other.ranges) && (ranges_partition == other.ranges_partition);
        }
    };

    label_resolution_map() = delete;
    explicit label_resolution_map(cell_labels_and_gids);

    const range_set& at(const cell_gid_type& gid, const cell_tag_type& tag) const {
        return map.at(gid).at(tag);
    }

    bool find(const cell_gid_type& gid, const cell_tag_type& tag) const {
        if (!map.count(gid)) return false;
        return map.at(gid).count(tag);
    }

    std::unordered_map<cell_gid_type, std::unordered_map<cell_tag_type, range_set>> map;
};

struct round_robin_state {
    cell_size_type state = 0;
    round_robin_state() : state(0) {};
    round_robin_state(cell_lid_type state) : state(state) {};
};

struct resolver {
    std::unordered_map<cell_gid_type, std::unordered_map<cell_tag_type, std::unordered_map <lid_selection_policy, round_robin_state>>> state_map;

    cell_lid_type resolve(const cell_global_label_type& iden, const label_resolution_map& label_map);
};
} // namespace arb
