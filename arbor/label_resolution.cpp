#include <vector>
#include <numeric>
#include <algorithm>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/util/hash_def.hpp>

#include "util/span.hpp"
#include "label_resolution.hpp"
#include "util/strprintf.hpp"

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
cell_lid_type range_set::at(unsigned idx) const {
    arb_assert(idx < size);
    for (const auto& [beg, end]: ranges) {
        auto len = end - beg;
        if (idx < len) return idx + beg;
        idx -= len;
    }
    ARB_UNREACHABLE
}

label_resolution_map::label_resolution_map(const cell_labels_and_gids& clg) {
    arb_assert(clg.label_range.check_invariant());
    const auto& gids = clg.gids;
    const auto& labels = clg.label_range.labels;
    const auto& ranges = clg.label_range.ranges;
    const auto& sizes = clg.label_range.sizes;

    singletons.reserve(labels.size());
    auto div = 0;
    for (auto gidx: util::count_along(gids)) {
        auto len = sizes[gidx];
        auto gid = gids[gidx];
        for (auto lidx: util::make_span(div, div + len)) {
            const auto& range = ranges[lidx];
            auto key = gid_label_pair{.gid=gid, .label=labels[lidx]};
            if (range.end  < range.begin) throw arb::arbor_internal_error("label_resolution_map: invalid lid_range");
            if (range.end == range.begin) continue;
            auto size = range.end - range.begin;
            // is a 'proper' range
            if (size > 1) {
                auto& rset = rangesets[key];
                rset.ranges.push_back(range);
                rset.size += size;
            }
            // already in rangesets
            else if (rangesets.contains(key)) {
                auto& rset = rangesets[key];
                rset.ranges.push_back(range);
                rset.size += size;
            }
            // key was already in singletons, so move to 'normal' rangesets and append
            else if (singletons.contains(key)) {
                auto& rset = rangesets[key];
                auto off = singletons[key];
                rset.ranges.push_back({off, off + 1});
                rset.ranges.push_back(range);
                rset.size = 1 + size; // remember to add the singleton's length
                singletons.erase(key);
            }
            // must be a pristine singleton
            else {
                singletons[key] = range.begin;
            }
        }
        div += len;
    }
}

std::size_t label_resolution_map::count(const cell_global_label_type& iden) { return count(iden.gid, iden.label.tag); }

std::size_t label_resolution_map::count(cell_gid_type gid, const cell_tag_type& label) {
    auto key = gid_label_pair{.gid=gid, .label=hash_value(label)};
    return singletons.count(key) + rangesets.count(key);
}

range_set label_resolution_map::at(const cell_global_label_type& iden) { return at(iden.gid, iden.label.tag); }

range_set label_resolution_map::at(cell_gid_type gid, const cell_tag_type& label) {
    auto key = gid_label_pair{.gid=gid, .label=hash_value(label)};
    if (auto it = singletons.find(key); it != singletons.end()) {
        return range_set{.size=1, .ranges={{it->second, it->second + 1}}};
    }
    if (auto it = rangesets.find(key); it != rangesets.end()) {
        return it->second;
    }
    throw std::range_error{util::pprintf("Key ({}, {}) not found.", gid, label)};
}

cell_lid_type resolver::resolve(const cell_global_label_type& iden) { return resolve(iden.gid, iden.label); }
cell_lid_type resolver::resolve(cell_gid_type gid, const cell_local_label_type& label) {
    // policy
    // 1) univalent        :: assert there's one target and return it
    // 2) round robin      :: return targets cyclically
    // 3) round robin halt :: return last target from 2), do not update counter
    const auto& [tag, pol] = label;
    // set up the key
    auto key = gid_label_pair{.gid=gid, .label=hash_value(tag)};
    // check whether our key points to a singleton (=range of length 1)
    // if so, the answer is always that one lid regardless of policy
    {
        const auto& it = label_map_->singletons.find(key);
        if (it != label_map_->singletons.end()) return it->second;
    }
    // was not a singleton, so look in the 'proper' ranges
    const auto& it = label_map_->rangesets.find(key);
    // fail, was neither in singletons nor here
    if (it == label_map_->rangesets.end()) throw arb::bad_connection_label(gid, tag, "label does not exist");
    // if it's not a singleton, univalent is invalid!
    if (pol == lid_selection_policy::assert_univalent) throw arb::bad_connection_label(gid, tag, "range is not univalent");
    // now policy must be rr or rr halt
    const auto& range_set = it->second;
    auto idx = rr_state_map_[key];
    // if rr, update counter, if rr halt don't.
    if (pol == lid_selection_policy::round_robin) rr_state_map_[key] = (idx + 1) % range_set.size;
    return range_set.at(idx);
}
} // namespace arb
