#pragma once

#include <unordered_map>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/hash_def.hpp>

#include "util/smallvec.hpp"

namespace arb {

// class containing the data required for {cell, label} to lid resolution.
// `sizes` is a partitioning vector for associating a cell with a set of
// (label, range) pairs in `labels`, `ranges`.
// gids of the cells are unknown.
struct ARB_ARBOR_API cell_label_range {
    cell_label_range() = default;
    cell_label_range(cell_label_range&&) = default;
    cell_label_range(const cell_label_range&) = default;
    cell_label_range& operator=(const cell_label_range&) = default;
    cell_label_range& operator=(cell_label_range&&) = default;

    cell_label_range(std::vector<cell_size_type> size_vec,
                     const std::vector<cell_tag_type>& label_vec,
                     std::vector<lid_range> range_vec);
    cell_label_range(std::vector<cell_size_type> size_vec,
                     std::vector<hash_type> label_vec,
                     std::vector<lid_range> range_vec);

    void add_cell();

    void add_label(hash_type label, lid_range range);

    void append(cell_label_range other);

    bool check_invariant() const;

    // The number of labels associated with each cell.
    std::vector<cell_size_type> sizes;
    // The labels corresponding to each cell, partitioned according to sizes_.
    std::vector<hash_type> labels;
    // The lid_range corresponding to each label.
    std::vector<lid_range> ranges;
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
struct ARB_ARBOR_API label_resolution_map {
    struct Key {
        uint64_t gid = 0;
        uint64_t label = 0;
        bool operator==(const Key&) const noexcept = default;
    };

    struct hasher {
        using is_avalanching = void;
        auto operator()(const Key& key) const noexcept {
            static_assert(std::has_unique_object_representations_v<Key>);
            return ankerl::unordered_dense::detail::wyhash::hash(&key, sizeof(key));
        }
    };

    struct range_set {
        std::size_t size = 0;
        std::size_t index = 0;
        std::size_t offset = 0;
        cell_lid_type at(unsigned idx, const std::vector<lid_range>& ranges) const;
    };

    template<typename V>
    using map_type = ankerl::unordered_dense::map<Key, V, hasher>;

    label_resolution_map() = default;
    explicit label_resolution_map(const cell_labels_and_gids&);

    const auto find(const Key& key) const { return map_.find(key); }
    const auto end() const { return map_.end(); }
    const range_set& at(const Key& key) const { return map_.at(key); }
    std::size_t count(const Key& key) const { return map_.count(key); }
    const range_set& at(cell_gid_type gid, hash_type hash) const { return at(Key(gid, hash)); }
    std::size_t count(cell_gid_type gid, hash_type hash) const { return count(Key(gid, hash)); }
    const range_set& at(cell_gid_type gid, const cell_tag_type& tag) const { return at(gid, hash_value(tag)); }
    std::size_t count(cell_gid_type gid, const cell_tag_type& tag) const { return count(gid, hash_value(tag)); }

    map_type<range_set> map_;
    std::vector<lid_range> ranges_;
};

// Struct used for resolving the lid of a (gid, label, lid_selection_policy) input.
// Requires a `label_resolution_map` which stores the constant mapping of (gid, label) pairs to lid sets.
struct ARB_ARBOR_API resolver {
    resolver() = delete;
    resolver(const label_resolution_map* label_map): label_map_(label_map) {}
    cell_lid_type resolve(const cell_global_label_type& iden);
    cell_lid_type resolve(cell_gid_type gid, const cell_local_label_type& lid);

    void clear() { rr_state_map_.clear(); }

private:
    const label_resolution_map* label_map_ = nullptr;
    // save index for round-robin and round-robin-halt policies
    label_resolution_map::map_type<cell_lid_type> rr_state_map_;
};
} // namespace arb
