#include <iterator>
#include <vector>
#include <numeric>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/util/hash_def.hpp>

#include "label_resolution.hpp"
#include "util/partition.hpp"
#include "util/span.hpp"

namespace arb {

// cell_label_range methods
cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   const std::vector<cell_tag_type>& label_vec,
                                   std::vector<lid_range> range_vec):
    sizes(std::move(size_vec)), ranges(std::move(range_vec)) {
    std::transform(label_vec.begin(), label_vec.end(),
                   std::back_inserter(labels),
                   hash_value<const std::string&>);
    arb_assert(check_invariant());
};

cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   std::vector<hash_type> label_vec,
                                   std::vector<lid_range> range_vec):
    sizes(std::move(size_vec)), labels(std::move(label_vec)), ranges(std::move(range_vec)) {
    arb_assert(check_invariant());
};


void cell_label_range::add_cell() { sizes.push_back(0); }

void cell_label_range::add_label(hash_type label, lid_range range) {
    if (sizes.empty()) throw arbor_internal_error("adding label to cell_label_range without cell");
    ++sizes.back();
    labels.push_back(label);
    ranges.push_back(std::move(range));
}

void cell_label_range::append(cell_label_range other) {
    using std::make_move_iterator;
    sizes.insert(sizes.end(),   make_move_iterator(other.sizes.begin()),  make_move_iterator(other.sizes.end()));
    labels.insert(labels.end(), make_move_iterator(other.labels.begin()), make_move_iterator(other.labels.end()));
    ranges.insert(ranges.end(), make_move_iterator(other.ranges.begin()), make_move_iterator(other.ranges.end()));
}

bool cell_label_range::check_invariant() const {
    const cell_size_type count = std::accumulate(sizes.begin(), sizes.end(), cell_size_type(0));
    return count==labels.size() && count==ranges.size();
}

// cell_labels_and_gids methods
cell_labels_and_gids::cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gid):
    label_range(std::move(lr)), gids(std::move(gid)) {
    if (label_range.sizes.size()!=gids.size()) throw arbor_internal_error("cell_label_range and gid count mismatch");
}

void cell_labels_and_gids::append(cell_labels_and_gids other) {
    label_range.append(other.label_range);
    gids.insert(gids.end(), make_move_iterator(other.gids.begin()), make_move_iterator(other.gids.end()));
}

bool cell_labels_and_gids::check_invariant() const {
    return label_range.check_invariant() && label_range.sizes.size()==gids.size();
}

/* The n-th local item (by index) to its identifier (lid). The lids are organised
   in potentially discontiguous ranges.

   idx --------           len0 <= idx < len0 + len1
                \
                 v
   | [s0, e0) [s1, e1), ... [sk, ek), ... |
       len0     len1
*/
cell_lid_type label_resolution_map::range_set::at(unsigned idx) const {
    if (0 == size) throw arbor_internal_error("no valid lids");
    for (const auto& [beg, end]: ranges) {
        auto len = end - beg;
        if (idx < len) return idx + beg;
        idx -= len;
    }
    throw arbor_internal_error("invalid lid");
}

label_resolution_map::label_resolution_map(const cell_labels_and_gids& clg) {
    arb_assert(clg.label_range.check_invariant());
    const auto& gids = clg.gids;
    const auto& labels = clg.label_range.labels;
    const auto& ranges = clg.label_range.ranges;
    const auto& sizes = clg.label_range.sizes;

    auto div = 0;
    for (auto gidx: util::count_along(gids)) {
        auto gid = gids[gidx];
        auto size = sizes[gidx];
        for (auto lidx: util::make_span(div, div + size)) {
            const auto& range = ranges[lidx];
            auto size = int(range.end - range.begin);
            if (size < 0) throw arb::arbor_internal_error("label_resolution_map: invalid lid_range");
            if (size == 0) continue;
            auto label = labels[lidx];
            auto& range_set = map[Key(gid, label)];
            range_set.ranges.push_back(range);
            range_set.size += size;
        }
        div += size;
    }
}

cell_lid_type resolver::resolve(const cell_global_label_type& iden) { return resolve(iden.gid, iden.label); }

cell_lid_type resolver::resolve(cell_gid_type gid, const cell_local_label_type& label) {
    const auto& [tag, pol] = label;
    auto key = label_resolution_map::Key(gid, hash_value(tag));
    auto it = label_map_->find(key);
    if (it == label_map_->end()) throw arb::bad_connection_label(gid, tag, "label does not exist");
    const auto& range_set = it->second;
    if (range_set.size <= 0) throw arb::bad_connection_label(gid, tag, "no valid lids");
    auto idx = 0;
    if (pol == lid_selection_policy::assert_univalent) {
        // must have single-entry range_set
        if (range_set.size != 1) throw arb::bad_connection_label(gid, tag, "range is not univalent");
    }
    else if (pol == lid_selection_policy::round_robin) {
        // cycle through range_set
        idx = rr_state_map_[key];
        rr_state_map_[key] = (idx + 1) % range_set.size;
    }
    else if (pol == lid_selection_policy::round_robin_halt) {
        // use previous state of round_robin policy
        idx = rr_state_map_[key];
    }
    return range_set.at(idx);
}
} // namespace arb
