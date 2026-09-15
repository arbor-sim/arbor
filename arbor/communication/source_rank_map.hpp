#pragma once

#include <numeric>
#include <vector>
#include <unordered_map>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

#include "communication/gathered_vector.hpp"

namespace arb {
struct sources_to_target_ranks {
    cell_size_type num_domains = 0;
    std::unordered_map<cell_member_type, std::vector<cell_size_type>> source_to_ranks;

    auto insert(const cell_member_type& key, cell_size_type val) { return source_to_ranks[key].push_back(val); }
    auto find_source(const cell_member_type& key) const { return source_to_ranks.find(key); }
    auto begin() const { return source_to_ranks.begin(); }
    auto end() const { return source_to_ranks.end(); }

    gathered_vector<spike>
    generate_all_to_all_vector(const std::vector<spike>& spikes) const {
        using count_type = gathered_vector<spike>::count_type;
        // count outgoing spikes per rank
        std::vector<count_type> offsets(num_domains + 1, 0);
        for (const auto& spk: spikes) {
            auto ranks = find_source(spk.source);
            if (ranks == end()) continue; 
            for (auto rank: ranks->second) {
                ++offsets[rank + 1];
            }
        }

        // make partition so we can sort the spikes into bins
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        // total number of spikes
        auto size = offsets.back();

        // we have the sizes per rank to send to, so deal spikes into bins.
        std::vector<spike> spikes_per_rank(size);
        auto rank_indices = offsets;
        for (const auto& spk: spikes) {
            auto ranks = find_source(spk.source);
            if (ranks == end()) continue;
            for (auto rank: ranks->second) {
                auto& index = rank_indices[rank];
                spikes_per_rank[index] = spk;
                ++index;
            }
        }
        return {std::move(spikes_per_rank), std::move(offsets)};
    }

    void reset() {
        num_domains = 0;
        source_to_ranks.clear();
    }
    
};

} // arb
