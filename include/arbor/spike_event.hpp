#pragma once

#include <tuple>
#include <vector>

#include <arbor/common_types.hpp>

namespace arb {

// Events delivered to targets on cells with a cell group.

struct spike_event {
    cell_member_type target;
    time_type time;
    float weight;

    friend bool operator==(const postsynaptic_spike_event& l, const postsynaptic_spike_event& r) {
        return l.target==r.target && l.time==r.time && l.weight==r.weight;
    }

    friend bool operator<(const postsynaptic_spike_event& l, const postsynaptic_spike_event& r) {
        return std::tie(l.time, l.target, l.weight) < std::tie(r.time, r.target, r.weight);
    }
};

using pse_vector = std::vector<spike_event>;

} // namespace arb
