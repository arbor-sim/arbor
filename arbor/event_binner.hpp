#pragma once

#include <limits>
#include <unordered_map>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>
#include <arbor/util/optional.hpp>

namespace arb {

class event_binner {
public:
    event_binner(): policy_(binning_kind::none), bin_interval_(0) {}

    event_binner(binning_kind policy, time_type bin_interval):
        policy_(policy), bin_interval_(bin_interval)
    {}

    void reset();

    // Determine binned time for an event based on policy.
    // If `t_min` is specified, the binned time will be no lower than `t_min`.
    // Otherwise the returned binned time will be less than or equal to the parameter `t`,
    // and within `bin_interval_`.

    time_type bin(time_type t, time_type t_min = std::numeric_limits<time_type>::lowest());

private:
    binning_kind policy_;

    // Interval in which event times can be aliased.
    time_type bin_interval_;

    util::optional<time_type> last_event_time_;
};

} // namespace arb

