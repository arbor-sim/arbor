#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

#include "event_binner.hpp"

namespace arb {

void event_binner::reset() {
    last_event_time_ = std::nullopt;
}

time_type event_binner::bin(time_type t, time_type t_min) {
    time_type t_binned = t;

    switch (policy_) {
    case binning_kind::none:
        break;
    case binning_kind::regular:
        if (bin_interval_>0) {
            t_binned = std::floor(t/bin_interval_)*bin_interval_;
        }
        break;
    case binning_kind::following:
        if (last_event_time_) {
            if (t-*last_event_time_<bin_interval_) {
                t_binned = *last_event_time_;
            }
        }
        last_event_time_ = t_binned;
        break;
    default:
        throw arbor_internal_error("event_binner: unrecognized binning policy");
    }

    return std::max(t_binned, t_min);
}

} // namespace arb

