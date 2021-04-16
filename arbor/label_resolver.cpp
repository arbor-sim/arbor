#include <vector>

#include <arbor/assert.hpp>
#include <arbor/label_resolver.hpp>
#include <arbor/common_types.hpp>

namespace arb {

cell_labeled_ranges::cell_labeled_ranges(std::vector<cell_gid_type> gids,
                                       std::vector<cell_tag_type> lbls,
                                       std::vector<lid_range> rngs):
    gids(std::move(gids)), labels(std::move(lbls)), ranges(std::move(rngs)) {};

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
}

void cell_labeled_ranges::append(cell_labeled_ranges other) {
    std::move(other.gids.begin(), other.gids.end(), std::back_inserter(gids));
    std::move(other.labels.begin(), other.labels.end(), std::back_inserter(labels));
    std::move(other.ranges.begin(), other.ranges.end(), std::back_inserter(ranges));
}

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_gid_range(cell_gid_type gid) const {
    auto it_0 = std::lower_bound(gids.begin(), gids.end(), gid);
    if (*it_0 != gid) return std::nullopt;
    auto it_1  = std::upper_bound(gids.begin(), gids.end(), gid);
    return std::make_pair(it_0 - gids.begin(), it_1 - gids.begin());
}

std::optional<std::pair<std::size_t, std::size_t>> cell_labeled_ranges::get_label_range(cell_tag_type label, std::pair<std::size_t, std::size_t> gid_range) const {
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

