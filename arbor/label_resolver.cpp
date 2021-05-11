#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "label_resolver.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"

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

    std::vector<cell_size_type> label_partition;
    util::make_partition(label_partition, clr.sizes);
    for(unsigned i = 0; i < clr.gids.size(); ++i) {
        auto labels = util::subrange_view(clr.labels, label_partition[i], label_partition[i+1]);
        auto ranges = util::subrange_view(clr.ranges, label_partition[i], label_partition[i+1]);

        label_resolution_map m;
        for (unsigned label_idx = 0; label_idx < labels.size(); ++label_idx) {
            m.insert({labels[label_idx], {ranges[label_idx], 0u}});
        }
        mapper.insert({clr.gids[i], std::move(m)});
    }
}

std::vector<cell_lid_type> label_resolver::get_lid(const cell_global_label_type& iden) const {
    std::vector<cell_lid_type> lids;
    if (!mapper.count(iden.gid) || !mapper.at(iden.gid).count(iden.label.tag)) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }

    auto matching_labels = mapper[iden.gid].equal_range(iden.label.tag);
    for (auto it = matching_labels.first; it != matching_labels.second; ++it) {
        auto& range_idx_pair = it->second;

        auto range = range_idx_pair.first;
        int size = range.end - range.begin;

        if (size < 1) {
            throw arb::bad_connection_range(iden.gid, iden.label.tag, range);
        }

        switch (iden.label.policy) {
        case lid_selection_policy::round_robin: {
            auto idx = range_idx_pair.second;
            range_idx_pair.second = (idx + 1) % size;
            lids.push_back(idx + range.begin);
            break;
        }
        case lid_selection_policy::assert_univalent: {
            if (size != 1) {
                throw arb::bad_univalent_connection_label(iden.gid, iden.label.tag);
            }
            lids.push_back(range.begin);
            break;
        }
        default: throw arb::bad_connection_label(iden.gid, iden.label.tag);
        }
    }
    return lids;
}

void label_resolver::reset() {
    for (auto& [gid, map]: mapper) {
        for (auto& [lable, range_idx_pair]: map) {
            range_idx_pair.second = 0;
        }
    }
}

} // namespace arb

