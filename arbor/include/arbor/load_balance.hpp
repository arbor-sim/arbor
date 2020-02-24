#pragma once

#include <set>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

struct cell_group_hint {
    constexpr static std::size_t max_size = -1;

    std::size_t cpu_group_size = 1;
    std::size_t gpu_group_size = max_size;
    bool prefer_gpu = true;
};

using gid_range_hint =  std::pair<cell_gid_type, cell_gid_type>;

struct custom_compare {
    bool operator() (const gid_range_hint& lhs, const gid_range_hint& rhs) const {
        return lhs.first < rhs.first;
    }
};

using cell_group_hint_map = std::unordered_map<cell_kind, cell_group_hint>;
using gid_range_hint_set  = std::set<gid_range_hint, custom_compare>;

struct partition_hint {
    cell_group_hint_map cell_group_map;
    gid_range_hint_set gid_range_set;

    void verify_gid_ranges(cell_gid_type num_cells) {
        std::vector<gid_range_hint> missing_ranges;
        cell_gid_type prev_range_end = 0;

        for (auto hint: gid_range_set) {
            if (hint.first < prev_range_end) {
                throw gid_range_check_failure("overlapping ranges");
            }
            if (hint.second > prev_range_end) {
                missing_ranges.push_back({prev_range_end, hint.first});
            }
            prev_range_end = hint.second;
        }
        if (prev_range_end > num_cells) {
            throw gid_range_check_failure("range outside total number of cells");
        }
        if (prev_range_end < num_cells) {
            gid_range_set.insert({prev_range_end, num_cells});
        }
        gid_range_set.insert(missing_ranges.begin(), missing_ranges.end());
    }
};

domain_decomposition partition_load_balance(
    const recipe& rec,
    const context& ctx,
    partition_hint hint_map = partition_hint());

} // namespace arb
