#pragma once

#include <unordered_map>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/expected.hpp>

#include "util/partition.hpp"

namespace arb {

using lid_hopefully = arb::util::expected<cell_lid_type, std::string>;

// class containing the data required for {cell, label} to lid resolution.
// `sizes` is a partitioning vector for associating a cell with a set of
// (label, range) pairs in `labels`, `ranges`.
// gids of the cells are unknown.
class ARB_ARBOR_API cell_label_range {
public:
    cell_label_range() = default;
    cell_label_range(cell_label_range&&) = default;
    cell_label_range(const cell_label_range&) = default;
    cell_label_range& operator=(const cell_label_range&) = default;
    cell_label_range& operator=(cell_label_range&&) = default;

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

// Struct for associating each cell of `cell_label_range` with a gid.
struct ARB_ARBOR_API cell_labels_and_gids {
    cell_labels_and_gids() = default;
    cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gids);

    void append(cell_labels_and_gids other);

    bool check_invariant() const;

    cell_label_range label_range;
    std::vector<cell_gid_type> gids;
};

// Class constructed from `cell_labels_and_ranges`:
// Represents the information in the object in a more
// structured manner for lid resolution in `resolver`
class ARB_ARBOR_API label_resolution_map {
public:
    struct range_set {
        std::vector<lid_range> ranges;
        std::vector<unsigned> ranges_partition = {0};
        cell_size_type size() const;
        lid_hopefully at(unsigned idx) const;
    };

    label_resolution_map() = default;
    explicit label_resolution_map(const cell_labels_and_gids&);

    const range_set& at(const cell_gid_type& gid, const cell_tag_type& tag) const;
    std::size_t count(const cell_gid_type& gid, const cell_tag_type& tag) const;

private:
    std::unordered_map<cell_gid_type, std::unordered_map<cell_tag_type, range_set>> map;
};

struct ARB_ARBOR_API round_robin_state {
    cell_lid_type state = 0;
    round_robin_state() : state(0) {};
    round_robin_state(cell_lid_type state) : state(state) {};
    cell_lid_type get();
    lid_hopefully update(const label_resolution_map::range_set& range);
};

struct ARB_ARBOR_API round_robin_halt_state {
    cell_lid_type state = 0;
    round_robin_halt_state() : state(0) {};
    round_robin_halt_state(cell_lid_type state) : state(state) {};
    cell_lid_type get();
    lid_hopefully update(const label_resolution_map::range_set& range);
};

struct ARB_ARBOR_API assert_univalent_state {
    cell_lid_type get();
    lid_hopefully update(const label_resolution_map::range_set& range);
};

// Struct used for resolving the lid of a (gid, label, lid_selection_policy) input.
// Requires a `label_resolution_map` which stores the constant mapping of (gid, label) pairs to lid sets.
struct ARB_ARBOR_API resolver {
    resolver(const label_resolution_map* label_map): label_map_(label_map) {}
    cell_lid_type resolve(const cell_global_label_type& iden);

private:
    using state_variant = std::variant<round_robin_state, round_robin_halt_state, assert_univalent_state>;

    state_variant construct_state(lid_selection_policy pol);
    state_variant construct_state(lid_selection_policy pol, cell_lid_type state);

    const label_resolution_map* label_map_;
    std::unordered_map<cell_gid_type, std::unordered_map<cell_tag_type, std::unordered_map <lid_selection_policy, state_variant>>> state_map_;
};
} // namespace arb
