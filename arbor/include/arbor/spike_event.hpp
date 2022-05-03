#pragma once

#include <iosfwd>
#include <tuple>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>

namespace arb {

// Events delivered to targets on cells with a cell group.

struct spike_event {
    cell_lid_type target;
    time_type time;
    float weight;

    friend bool operator==(const spike_event& l, const spike_event& r) {
        return l.target==r.target && l.time==r.time && l.weight==r.weight;
    }

    friend bool operator<(const spike_event& l, const spike_event& r) {
        return std::tie(l.time, l.target, l.weight) < std::tie(r.time, r.target, r.weight);
    }
};

using pse_vector = std::vector<spike_event>;

struct cell_spike_events {
    cell_gid_type target;
    pse_vector events;
};

using cse_vector = std::vector<cell_spike_events>;

ARB_ARBOR_API std::ostream& operator<<(std::ostream&, const spike_event&);

} // namespace arb
