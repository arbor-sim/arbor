#pragma once

#include <arbor/arb_types.hpp>

#include <iosfwd>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/serdes.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/lexcmp_def.hpp>

namespace arb {

// Events delivered to targets on cells with a cell group.

struct spike_event {
    cell_lid_type target = -1;
    float weight = 0;
    time_type time = -1;

    spike_event() = default;
    constexpr spike_event(cell_lid_type tgt, time_type t, arb_weight_type w) noexcept: target(tgt), weight(w), time(t) {}

    ARB_SERDES_ENABLE(spike_event, target, time, weight);
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(spike_event,(a.time,a.target,a.weight),(b.time,b.target,b.weight))

using pse_vector = std::vector<spike_event>;

struct cell_spike_events {
    cell_gid_type target;
    pse_vector events;
};

using cse_vector = std::vector<cell_spike_events>;

ARB_ARBOR_API std::ostream& operator<<(std::ostream&, const spike_event&);

} // namespace arb
