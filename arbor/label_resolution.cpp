#include <iterator>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "label_resolution.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

// cell_label_range methods
cell_label_range::cell_label_range(std::vector<cell_size_type> size_vec,
                                   std::vector<cell_tag_type> label_vec,
                                   std::vector<lid_range> range_vec):
    sizes_(std::move(size_vec)), labels_(std::move(label_vec)), ranges_(std::move(range_vec))
{
    arb_assert(check_invariant());
};

void cell_label_range::add_cell() {
    sizes_.push_back(0);
}

void cell_label_range::add_label(cell_tag_type label, lid_range range) {
    if (sizes_.empty()) throw arbor_internal_error("adding label to cell_label_range without cell");
    ++sizes_.back();
    labels_.push_back(std::move(label));
    ranges_.push_back(std::move(range));
}

void cell_label_range::append(cell_label_range other) {
    using std::make_move_iterator;
    sizes_.insert(sizes_.end(), make_move_iterator(other.sizes_.begin()), make_move_iterator(other.sizes_.end()));
    labels_.insert(labels_.end(), make_move_iterator(other.labels_.begin()), make_move_iterator(other.labels_.end()));
    ranges_.insert(ranges_.end(), make_move_iterator(other.ranges_.begin()), make_move_iterator(other.ranges_.end()));
}

bool cell_label_range::check_invariant() const {
    const cell_size_type count = std::accumulate(sizes_.begin(), sizes_.end(), cell_size_type(0));
    return count==labels_.size() && count==ranges_.size();
}

// cell_labels_and_gids methods
cell_labels_and_gids::cell_labels_and_gids(cell_label_range lr, std::vector<cell_gid_type> gid):
    label_range(std::move(lr)), gids(std::move(gid))
{
    if (label_range.sizes().size()!=gids.size()) throw arbor_internal_error("cell_label_range and gid count mismatch");
}

void cell_labels_and_gids::append(cell_labels_and_gids other) {
    label_range.append(other.label_range);
    gids.insert(gids.end(), make_move_iterator(other.gids.begin()), make_move_iterator(other.gids.end()));
}

bool cell_labels_and_gids::check_invariant() const {
    return label_range.check_invariant() && label_range.sizes().size()==gids.size();
}

// label_resolution_map methods
unsigned label_resolution_map::range_set::size() const {
    return ranges_partition.back();
}

unsigned label_resolution_map::range_set::at(unsigned idx) const {
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

const label_resolution_map::range_set& label_resolution_map::at(const cell_gid_type& gid, const cell_tag_type& tag) const {
    return map.at(gid).at(tag);
}

bool label_resolution_map::find(const cell_gid_type& gid, const cell_tag_type& tag) const {
    if (!map.count(gid)) return false;
    return map.at(gid).count(tag);
}

label_resolution_map::label_resolution_map(const cell_labels_and_gids& clg) {
    arb_assert(clg.label_range.check_invariant());
    const auto& gids = clg.gids;
    const auto& labels = clg.label_range.labels();
    const auto& ranges = clg.label_range.ranges();
    const auto& sizes = clg.label_range.sizes();

    std::vector<cell_size_type> label_divs;
    auto partn = util::make_partition(label_divs, sizes);
    for (auto i: util::count_along(partn)) {
        auto gid = gids[i];

        std::unordered_map<cell_tag_type, range_set> m;
        for (auto label_idx: util::make_span(partn[i])) {
            auto& range_set = m[labels[label_idx]];

            const auto range = ranges[label_idx];
            auto size = int(range.end - range.begin);

            if (size < 0) {
                throw arb::bad_connection_range(gids[i], labels[label_idx], range);
            }

            range_set.ranges.push_back(range);
            range_set.ranges_partition.push_back(range_set.ranges_partition.back() + size);
        }

        for (const auto& [label, range_set]: m) {
            if (range_set.ranges_partition.back() < 1) {
                throw arb::bad_connection_set(gid, label);
            }
        }
        map.insert({gid, std::move(m)});
    }
}

// resolver methods
cell_lid_type resolver::resolve(const cell_global_label_type& iden, const label_resolution_map& label_map) {
    if (!label_map.find(iden.gid, iden.label.tag)) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }
    const auto& range_set = label_map.at(iden.gid, iden.label.tag);

    switch (iden.label.policy) {
    case lid_selection_policy::round_robin: {
        // Get the state of the round_robin iterator.
        auto& label_states = state_map[iden.gid][iden.label.tag];
        const auto curr = !label_states.count(iden.label.policy) ? round_robin_state(0) : label_states.at(iden.label.policy);

        // Update the state of the round_robin iterator.
        label_states[iden.label.policy] = round_robin_state((curr.state+1) % range_set.size());

        // Get the lid at the current index.
        return range_set.at(curr.state);
    }
    case lid_selection_policy::assert_univalent: {
        if (range_set.size() != 1) {
            throw arb::bad_univalent_connection_label(iden.gid, iden.label.tag);
        }
        // Get the lid of the only element.
        return range_set.at(0);
    }
    default: throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }
}

} // namespace arb

