#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "label_resolver.hpp"
#include "util/partition.hpp"

namespace arb {

cell_labeled_ranges::cell_labeled_ranges(std::vector<cell_gid_type> ids,
                                         std::vector<cell_tag_type> lbls,
                                         std::vector<lid_range> rngs,
                                         std::vector<std::size_t> ptns):
    gids(std::move(ids)), labels(std::move(lbls)), ranges(std::move(rngs)), sorted_partitions(std::move(ptns))
{
    arb_assert(labels.size() == gids.size());
    arb_assert(ranges.size() == gids.size());
};

cell_labeled_ranges::cell_labeled_ranges(const clr_vector& tuple_vec) {
    arb_assert(std::is_sorted(tuple_vec.begin(), tuple_vec.end()));
    gids.reserve(tuple_vec.size());
    labels.reserve(tuple_vec.size());
    ranges.reserve(tuple_vec.size());
    for (const auto& item: tuple_vec) {
        gids.push_back(std::get<0>(item));
        labels.push_back(std::get<1>(item));
        ranges.push_back(std::get<2>(item));
    }
    sorted_partitions.push_back(0);
    sorted_partitions.push_back(tuple_vec.size());
}

bool cell_labeled_ranges::is_one_partition() const {
    return (sorted_partitions.size()==2 && sorted_partitions.front()== 0 && sorted_partitions.back()==gids.size());
};

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_gid_range(cell_gid_type gid, int partition) const {
    auto part = (partition == -1) ? std::make_pair(sorted_partitions.front(), sorted_partitions.back()) : util::partition_view(sorted_partitions).at(partition);
    auto first = gids.begin()+part.first;
    auto last  = gids.begin()+part.second;
    auto it_0 = std::lower_bound(first, last, gid);
    if (it_0 == gids.end() || *it_0 != gid) {
        return std::nullopt;
    }
    auto it_1 = std::upper_bound(first, last, gid);
    return std::make_pair(it_0 - gids.begin(), it_1 - gids.begin());
}

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_label_range(const cell_tag_type& label, std::pair<std::size_t, std::size_t> gid_range) const {
    auto first = labels.begin()+gid_range.first;
    auto last  = labels.begin()+gid_range.second;
    auto it_0 = std::lower_bound(first, last, label);
    if (it_0 == labels.end() || *it_0 != label) {
        return std::nullopt;
    }
    auto it_1  = std::upper_bound(first, last, label);
    return std::make_pair(it_0 - labels.begin(), it_1 - labels.begin());
}

label_resolver::label_resolver(cell_labeled_ranges ranges):
    mapper(std::move(ranges)),
    indices(mapper.gids.size(), 0) {
    arb_assert(mapper.labels.size() == indices.size());
    arb_assert(mapper.ranges.size() == indices.size());
}

cell_lid_type label_resolver::get_lid(cell_global_label_type iden, int rank) const {
    auto gid_range = mapper.get_gid_range(iden.gid, rank);
    if (!gid_range) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }
    auto label_range = mapper.get_label_range(iden.label.tag, gid_range.value());
    if (!label_range) {
        throw arb::bad_connection_label(iden.gid, iden.label.tag);
    }
    arb_assert(label_range.value().second - label_range.value().first == 1);

    auto rid = label_range.value().first;
    auto range = mapper.ranges[rid];
    auto size = range.end - range.begin;

    switch (iden.label.policy) {
    case lid_selection_policy::round_robin:
    {
        auto idx = indices[rid];
        indices[rid] = (idx+1)%size;
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
} // namespace arb

