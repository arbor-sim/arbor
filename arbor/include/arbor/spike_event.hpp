#pragma once

#include <arbor/arb_types.hpp>

#include <iosfwd>
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

    bool operator==(const spike_event&) const = default;
    constexpr auto operator<=>(const spike_event& o) const { return std::tie(time, target, weight) <=> std::tie(o.time, o.target, o.weight); }

    ARB_SERDES_ENABLE(spike_event, target, time, weight);
};

using pse_vector = std::vector<spike_event>;

ARB_ARBOR_API std::ostream& operator<<(std::ostream&, const spike_event&);

} // namespace arb
