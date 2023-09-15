#pragma once

#include <arbor/arb_types.hpp>

#include <iosfwd>
#include <tuple>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/serdes.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Events delivered to targets on cells with a cell group.

struct spike_event {
    cell_lid_type target = -1;
    float weight = 0;
    time_type time = -1;

    spike_event() = default;
    constexpr spike_event(cell_lid_type tgt, time_type t, arb_weight_type w) noexcept: target(tgt), weight(w), time(t) {}

    friend bool operator==(const spike_event& l, const spike_event& r) {
        return l.target==r.target && l.time==r.time && l.weight==r.weight;
    }

    friend bool operator<(const spike_event& l, const spike_event& r) {
        return std::tie(l.time, l.target, l.weight) < std::tie(r.time, r.target, r.weight);
    }

    ARB_SERDES_ENABLE(spike_event, target, time, weight);
};

using pse_vector = std::vector<spike_event>;

struct cell_spike_events {
    cell_gid_type target;
    pse_vector events;
};

using cse_vector = std::vector<cell_spike_events>;

ARB_ARBOR_API std::ostream& operator<<(std::ostream&, const spike_event&);

} // namespace arb
