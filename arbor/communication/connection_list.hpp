#pragma once

#include <arbor/common_types.hpp>

#include <ankerl/unordered_dense.h>

#include "util/rangeutil.hpp"
#include "connection.hpp"

namespace arb {
struct connections_list {
    size_t size_ = 0;
    std::vector<cell_size_type> idx_on_domain;
    std::vector<cell_lid_type> dests;
    std::vector<cell_member_type> srcs;
    std::vector<float> weights;
    std::vector<float> delays;
    // for each domain, map sources to ranges of connctions.
    std::vector<ankerl::unordered_dense::map<std::uint64_t, std::pair<std::size_t, std::size_t>>> first_occurence;

    void make(std::vector<connection>& cons) {
        arb_assert(util::is_sorted(cons));
        first_occurence.emplace_back();
        auto& lut = first_occurence.back();
        for (const auto& con: cons) {
            auto key = std::bit_cast<std::uint64_t>(con.source);
            if (!lut.contains(key)) lut.emplace(key, std::make_pair(size_, size_));
            lut[key].second += 1;
            idx_on_domain.push_back(con.index_on_domain);
            dests.push_back(con.target);
            srcs.push_back(con.source);
            weights.push_back(con.weight);
            delays.push_back(con.delay);
            ++size_;
        }
    }

    void make(std::vector<std::vector<connection>>& conss) {
        for (auto& cons: conss) {
            make(cons);
            // NOTE: For memory capacity reasons, we destroy
            //       the sub-vectors here, once we are done.
            cons = {};
        }
    }

    void reserve(std::size_t n) {
        idx_on_domain.reserve(n);
        dests.reserve(n);
        weights.reserve(n);
        delays.reserve(n);
    }

    void clear() {
        srcs.clear();
        idx_on_domain.clear();
        dests.clear();
        weights.clear();
        delays.clear();
        size_ = 0;
    }

    size_t size() const { return size_; }
};
}
