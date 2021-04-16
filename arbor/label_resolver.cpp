#include <vector>

#include <arbor/assert.hpp>
#include <arbor/label_resolver.hpp>
#include <arbor/common_types.hpp>

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

cell_labeled_ranges::cell_labeled_ranges(const std::vector<std::tuple<cell_gid_type, std::string, lid_range>>& tuple_vec) {
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

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_gid_range(cell_gid_type gid) const {
    for (const auto& part : util::partition_view(sorted_partitions)) {
        auto it_0 = std::lower_bound(gids.begin()+part.first, gids.end()+part.second, gid);
        if (*it_0 != gid) continue;
        auto it_1 = std::upper_bound(gids.begin(), gids.end(), gid);
        return std::make_pair(it_0 - gids.begin(), it_1 - gids.begin());
    }
    return std::nullopt;
}

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_label_range(const cell_tag_type& label, std::pair<std::size_t, std::size_t> gid_range) const {
    auto it_0 = std::lower_bound(labels.begin()+gid_range.first, labels.begin()+gid_range.second, label);
    if (*it_0 != label) return std::nullopt;
    auto it_1  = std::upper_bound(labels.begin(), labels.end(), label);
    return std::make_pair(it_0 - labels.begin(), it_1 - labels.begin());
}

label_resolver::label_resolver(cell_labeled_ranges ranges):
    mapper(std::move(ranges)),
    indices(mapper.gids.size(), 0) {
    arb_assert(mapper.labels.size() == indices.size());
    arb_assert(mapper.ranges.size() == indices.size());
    for (unsigned i = 0; i < ranges.gids.size(); ++i) {
        std::cout << ranges.gids[i] << ", " << ranges.labels[i] << ", (" << ranges.ranges[i].begin << " -> " << ranges.ranges[i].end << ")" << std::endl;
    }
    std::cout << "[";
    for (auto i:ranges.sorted_partitions) {
        std::cout << i << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
}

std::optional<cell_lid_type> label_resolver::get_lid(const cell_label_type& elem, lid_selection_policy policy) const {
    auto gid_range = mapper.get_gid_range(elem.gid);
    if(!gid_range) return std::nullopt;

    auto label_range = mapper.get_label_range(elem.label, gid_range.value());
    if(!label_range) return std::nullopt;

    arb_assert(label_range.value().first == label_range.value().second);

    auto rid = label_range.value().first;
    auto range = mapper.ranges[rid];
    auto size = range.end - range.begin;

    switch (policy) {
    case lid_selection_policy::round_robin:
    {
        auto idx = indices[rid];
        indices[idx] = (idx+1)%size;
        return idx + range.begin;
    }
    case lid_selection_policy::assert_univalent:
    {
        if (size != 1) {
            return std::nullopt;
        }
        return range.begin;
    }
    }
    return std::nullopt;
}
} // namespace arb

