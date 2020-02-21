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

struct gid_range_hint {
    std::pair<cell_gid_type, cell_gid_type> gid_range;
    unsigned complexity;
};

struct partition_hint {
    struct custom_compare {
        bool operator() (const gid_range_hint& lhs, const gid_range_hint& rhs) const {
            return lhs.gid_range.first < rhs.gid_range.first;
        }
    };

    std::unordered_map<cell_kind, cell_group_hint> cell_group_hint_map;
    std::set<gid_range_hint, custom_compare> gid_range_hint_set;

    void verify_gid_ranges(cell_gid_type num_cells) {
        std::vector<gid_range_hint> missing_ranges;
        cell_gid_type prev_range_end = 0;
        for (auto hint: gid_range_hint_set) {
            if (hint.gid_range.first < prev_range_end) {
                throw gid_range_check_failure("overlapping ranges");
            }
            if (hint.gid_range.second > prev_range_end) {
                missing_ranges.push_back({{prev_range_end, hint.gid_range.first}, 0});
            }
            prev_range_end = hint.gid_range.second;
        }
        if (prev_range_end > num_cells) {
            throw gid_range_check_failure("range outside total number of cells");
        }
        if (prev_range_end < num_cells) {
            gid_range_hint_set.insert({{prev_range_end, num_cells}, 0});
        }
        gid_range_hint_set.insert(missing_ranges.begin(), missing_ranges.end());
    }
};

domain_decomposition partition_load_balance(
    const recipe& rec,
    const context& ctx,
    partition_hint hint_map = {});

} // namespace arb
