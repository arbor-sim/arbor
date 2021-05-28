#include <iterator>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "label_resolver.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {

cell_label_range::cell_label_range(std::vector<cell_gid_type> gid_vec,
                                   std::vector<cell_size_type> size_vec,
                                   std::vector<cell_tag_type> label_vec,
                                   std::vector<lid_range> range_vec):
    gids(std::move(gid_vec)), sizes(std::move(size_vec)), labels(std::move(label_vec)), ranges(std::move(range_vec))
{
    arb_assert(labels.size() == ranges.size());
    arb_assert(gids.size() == sizes.size());
};

void cell_label_range::append(cell_label_range other) {
    gids.insert(gids.end(), std::make_move_iterator(other.gids.begin()), std::make_move_iterator(other.gids.end()));
    sizes.insert(sizes.end(), std::make_move_iterator(other.sizes.begin()), std::make_move_iterator(other.sizes.end()));
    labels.insert(labels.end(), std::make_move_iterator(other.labels.begin()), std::make_move_iterator(other.labels.end()));
    ranges.insert(ranges.end(), std::make_move_iterator(other.ranges.begin()), std::make_move_iterator(other.ranges.end()));
}

label_resolver::label_resolver(cell_label_range clr) {
    arb_assert(clr.gids.size() == clr.sizes.size());
    arb_assert(clr.labels.size() == clr.ranges.size());

    std::vector<cell_size_type> label_divs;
    auto partn = util::make_partition(label_divs, clr.sizes);
    for (auto i: util::count_along(partn)) {
        auto gid = clr.gids[i];

        label_resolution_map m;
        for (auto label_idx: util::make_span(partn[i])) {
            auto& pair = m[clr.labels[label_idx]];
            auto& const_state = pair.first;
            auto& mutable_state = pair.second;

            const auto range = clr.ranges[label_idx];
            const int size = range.end - range.begin;

            if (size < 0) {
                throw arb::bad_connection_range(clr.gids[i], clr.labels[label_idx], range);
            }

            const_state.ranges.push_back(range);
            const_state.ranges_partition.push_back(const_state.ranges_partition.back() + size);
            mutable_state = 0;
        }

        for (const auto& [label, state]: m) {
            if (state.first.ranges_partition.back() < 1) {
                throw arb::bad_connection_set(gid, label);
            }
        }
        mapper.insert({gid, std::move(m)});
    }
}

cell_lid_type label_resolver::get_lid(const cell_global_label_type& iden) const {
    if (!mapper.count(iden.gid) || !mapper.at(iden.gid).count(iden.label.tag)) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }

    auto& range_idx_pair = mapper[iden.gid][iden.label.tag];

    auto ranges = range_idx_pair.first.ranges;
    auto ranges_part = util::partition_view(range_idx_pair.first.ranges_partition);
    auto size = ranges_part.bounds().second;

    switch (iden.label.policy) {
    case lid_selection_policy::round_robin: {
        // Get the state of the round_robin iterator.
        auto curr = range_idx_pair.second;

        // Update the state of the round_robin iterator.
        range_idx_pair.second = (curr + 1) % size;

        // Get the range that contains the current state.
        auto range_idx = ranges_part.index(curr);
        auto range = range_idx_pair.first.ranges.at(range_idx);

        // Get the offset into that range
        auto offset = curr - range_idx_pair.first.ranges_partition.at(range_idx);

        return range.begin + offset;
    }
    case lid_selection_policy::assert_univalent: {
        if (size != 1) {
            throw arb::bad_univalent_connection_label(iden.gid, iden.label.tag);
        }
        // Get the range that contains the only element.
        auto range_idx = ranges_part.index(0);
        auto range = range_idx_pair.first.ranges.at(range_idx);
        return range.begin;
    }
    default: throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }
}

void label_resolver::reset() {
    for (auto& [gid, map]: mapper) {
        for (auto& [lable, range_idx_pair]: map) {
            range_idx_pair.second = 0;
        }
    }
}

} // namespace arb

