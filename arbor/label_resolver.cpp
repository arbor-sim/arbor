#include <vector>
#include <iostream>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "label_resolver.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"

namespace arb {

cell_labeled_ranges::cell_labeled_ranges(std::vector<cell_gid_type> gid_vec,
                                         std::vector<cell_size_type> size_vec,
                                         std::vector<cell_tag_type> label_vec,
                                         std::vector<lid_range> range_vec):
    gids(std::move(gid_vec)), sizes(std::move(size_vec)), labels(std::move(label_vec)), ranges(std::move(range_vec))
{
    arb_assert(labels.size() == ranges.size());
    arb_assert(gids.size() == sizes.size());
};

void cell_labeled_ranges::append(cell_labeled_ranges other) {
    gids.insert(gids.end(), std::make_move_iterator(other.gids.begin()), std::make_move_iterator(other.gids.end()));
    sizes.insert(sizes.end(), std::make_move_iterator(other.sizes.begin()), std::make_move_iterator(other.sizes.end()));
    labels.insert(labels.end(), std::make_move_iterator(other.labels.begin()), std::make_move_iterator(other.labels.end()));
    ranges.insert(ranges.end(), std::make_move_iterator(other.ranges.begin()), std::make_move_iterator(other.ranges.end()));
}

label_resolver::label_resolver(cell_labeled_ranges clr) {
    arb_assert(clr.gids.size() == clr.sizes.size());
    arb_assert(clr.labels.size() == clr.ranges.size());

    std::cout << "gid : ";
    for (auto a: clr.gids) {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    std::cout << "sizes : ";
    for (auto a: clr.sizes) {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    std::cout << "labels : ";
    for (auto a: clr.labels) {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    std::cout << "ranges : ";
    for (auto a: clr.ranges) {
        std::cout << "{" << a.begin << ", " << a.end <<"} ";
    }
    std::cout << std::endl<< std::endl;

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

cell_lid_type label_resolver::get_lid(const cell_global_label_type& iden) const {
    if (!mapper.count(iden.gid) || !mapper.at(iden.gid).count(iden.label.tag)) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }

    auto& range_idx_pair = mapper[iden.gid][iden.label.tag];

    auto range = range_idx_pair.first;
    auto size = range.end - range.begin;

    switch (iden.label.policy) {
    case lid_selection_policy::round_robin:
    {
        auto idx = range_idx_pair.second;
        range_idx_pair.second = (idx+1)%size;
        return idx + range.begin;
    }
    case lid_selection_policy::assert_univalent:
    {
        if (size != 1) {
            throw arb::bad_univalent_connection_label(iden.gid, iden.label.tag);
        }
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

