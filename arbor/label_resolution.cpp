#include <iterator>
#include <vector>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/expected.hpp>

#include "label_resolution.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

// cell_label_range methods
cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   std::vector<cell_tag_type> label_vec,
                                   std::vector<lid_range> range_vec):
    sizes(std::move(size_vec)), ranges(std::move(range_vec))
{
    std::transform(label_vec.begin(), label_vec.end(),
                   std::back_inserter(labels),
                   internal_hash);
    arb_assert(check_invariant());
};

void cell_label_range::add_cell() { sizes.push_back(0); }

void cell_label_range::add_label(cell_tag_type label, lid_range range) {
    if (sizes.empty()) throw arbor_internal_error("adding label to cell_label_range without cell");
    ++sizes.back();
    labels.push_back(internal_hash(label));
    ranges.push_back(std::move(range));
}

void cell_label_range::append(cell_label_range other) {
    using std::make_move_iterator;
    sizes.insert(sizes.end(), make_move_iterator(other.sizes.begin()), make_move_iterator(other.sizes.end()));
    labels.insert(labels.end(), make_move_iterator(other.labels.begin()), make_move_iterator(other.labels.end()));
    ranges.insert(ranges.end(), make_move_iterator(other.ranges.begin()), make_move_iterator(other.ranges.end()));
}

bool cell_label_range::check_invariant() const {
    const cell_size_type count = std::accumulate(sizes.begin(), sizes.end(), cell_size_type(0));
    size_t beg = 0;
    for (auto size: sizes) {
        size_t end = beg + size;
        std::unordered_set<hash_type> seen;
        for (auto idx = beg; idx < end; ++idx) {
            auto hash = labels[idx];
            if (seen.count(hash)) return false;
            seen.insert(hash);
        }
        beg = end;
    }
    return count==labels.size() && count==ranges.size();
}

// cell_labels_and_gids methods
cell_labels_and_gids::cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gid):
    label_range(std::move(lr)), gids(std::move(gid))
{
    if (label_range.sizes.size()!=gids.size()) throw arbor_internal_error("cell_label_range and gid count mismatch");
}

void cell_labels_and_gids::append(cell_labels_and_gids other) {
    label_range.append(other.label_range);
    gids.insert(gids.end(), make_move_iterator(other.gids.begin()), make_move_iterator(other.gids.end()));
}

bool cell_labels_and_gids::check_invariant() const {
    return label_range.check_invariant() && label_range.sizes.size()==gids.size();
}

// label_resolution_map methods
cell_size_type label_resolution_map::range_set::size() const {
    return ranges_partition.back();
}

lid_hopefully label_resolution_map::range_set::at(unsigned idx) const {
    if (size() < 1) return util::unexpected("no valid lids");
    auto part = util::partition_view(ranges_partition);
    // Index of the range containing idx.
    auto ridx = part.index(idx);

    // First element of the range containing idx.
    const auto& start = ranges.at(ridx).begin;

    // Offset into the range containing idx.
    const auto& range_part = part.at(ridx);
    auto offset = idx - range_part.first;
    return start + offset;
}

const label_resolution_map::range_set& label_resolution_map::at(cell_gid_type gid, const cell_tag_type& tag) const {
    return map.at(gid).at(internal_hash(tag));
}

std::size_t label_resolution_map::count(cell_gid_type gid, const cell_tag_type& tag) const {
    if (!map.count(gid)) return 0u;
    return map.at(gid).count(internal_hash(tag));
}

label_resolution_map::label_resolution_map(const cell_labels_and_gids& clg) {
    arb_assert(clg.label_range.check_invariant());
    const auto& gids = clg.gids;
    const auto& labels = clg.label_range.labels;
    const auto& ranges = clg.label_range.ranges;
    const auto& sizes = clg.label_range.sizes;

    std::vector<cell_size_type> label_divs;
    auto partn = util::make_partition(label_divs, sizes);
    for (auto i: util::count_along(partn)) {
        auto gid = gids[i];

        std::unordered_map<hash_type, range_set> m;
        for (auto label_idx: util::make_span(partn[i])) {
            const auto range = ranges[label_idx];
            auto size = int(range.end - range.begin);
            if (size < 0) {
                throw arb::arbor_internal_error("label_resolution_map: invalid lid_range");
            }
            auto& label = labels[label_idx];
            auto& range_set = m[label];
            range_set.ranges.push_back(range);
            range_set.ranges_partition.push_back(range_set.ranges_partition.back() + size);
        }
        if (!map.insert({gid, std::move(m)}).second) {
            throw arb::arbor_internal_error("label_resolution_map: duplicate gid");
        }
    }
}
// variant state methods
lid_hopefully round_robin_state::update(const label_resolution_map::range_set& range_set) {
    auto lid = range_set.at(state);
    if (lid) state = (state+1) % range_set.size();
    return lid;
}

cell_lid_type round_robin_state::get() {
    return state;
}

lid_hopefully round_robin_halt_state::update(const label_resolution_map::range_set& range_set) {
    return range_set.at(state);
}

cell_lid_type round_robin_halt_state::get() {
    return state;
}

lid_hopefully assert_univalent_state::update(const label_resolution_map::range_set& range_set) {
    if (range_set.size() != 1) {
        return util::unexpected("range is not univalent");
    }
    // Get the lid of the only element.
    return range_set.at(0);
}

cell_lid_type assert_univalent_state::get() {
    return 0;
}

// resolver methods
resolver::state_variant resolver::construct_state(lid_selection_policy pol) {
    switch (pol) {
    case lid_selection_policy::round_robin:
        return round_robin_state();
	case lid_selection_policy::round_robin_halt:
        return round_robin_halt_state();
    case lid_selection_policy::assert_univalent:
       return assert_univalent_state();
    default: return assert_univalent_state();
    }
}

resolver::state_variant resolver::construct_state(lid_selection_policy pol, cell_lid_type state) {
    switch (pol) {
    case lid_selection_policy::round_robin:
        return round_robin_state(state);
	case lid_selection_policy::round_robin_halt:
        return round_robin_halt_state(state);
    case lid_selection_policy::assert_univalent:
       return assert_univalent_state();
    default: return assert_univalent_state();
    }
}

cell_lid_type resolver::resolve(const cell_global_label_type& iden) { return resolve(iden.gid, iden.label); }

cell_lid_type get_state(resolver::state_variant& v) {
    if (std::holds_alternative<round_robin_state>(v)) {
        return std::get<round_robin_state>(v).get();
    }
    else if (std::holds_alternative<round_robin_halt_state>(v)) {
        return std::get<round_robin_halt_state>(v).get();
    }
    else {
        return std::get<assert_univalent_state>(v).get();
    }
}

lid_hopefully update_state(resolver::state_variant& v,
                           const label_resolution_map::range_set& r) {
    if (std::holds_alternative<round_robin_state>(v)) {
        return std::get<round_robin_state>(v).update(r);
    }
    else if (std::holds_alternative<round_robin_halt_state>(v)) {
        return std::get<round_robin_halt_state>(v).update(r);
    }
    else {
        return std::get<assert_univalent_state>(v).update(r);
    }
}

cell_lid_type resolver::resolve(cell_gid_type gid, const cell_local_label_type& label) {
    const auto& [tag, pol] = label;

    auto hash = internal_hash(tag);

    if (!label_map_->count(gid, tag)) throw arb::bad_connection_label(gid, tag, "label does not exist");
    const auto& range_set = label_map_->at(gid, tag);

    auto& state = state_map_[gid][hash];

    // Policy round_robin_halt: use previous state of round_robin policy, if existent
    if (pol == lid_selection_policy::round_robin_halt
     && state.count(lid_selection_policy::round_robin)) {
        auto prev_state_rr = get_state(state[lid_selection_policy::round_robin]);
        state[lid_selection_policy::round_robin_halt] = construct_state(lid_selection_policy::round_robin_halt,
                                                                        prev_state_rr);
    }

    // Construct state if it doesn't exist
    if (!state.count(pol)) state[pol] = construct_state(pol);

    // Update state
    auto lid = update_state(state[pol], range_set);

    if (!lid) throw arb::bad_connection_label(gid, tag, lid.error());
    return *lid;
}

} // namespace arb

