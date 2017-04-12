#pragma once

#include <cmath>
#include <limits>
#include <unordered_map>

#include <spike.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

enum class binning_kind {
    none,
    regular,   // => round time down to multiple of binning interval.
    following, // => round times down to previous event if within binning interval.
};

class event_binner {
public:
    using time_type = spike::time_type;

    event_binner(): policy_(binning_kind::none), bin_interval_(0) {}

    event_binner(binning_kind policy, time_type bin_interval):
        policy_(policy), bin_interval_(bin_interval)
    {}

    void reset();

    // Determine binned time for an event based on policy.
    // If `t_min` is specified, the binned time will be no lower than `t_min`.
    // Otherwise the returned binned time will be less than or equal to the parameter `t`,
    // and within `bin_interval_`.

    time_type bin(cell_gid_type id,
                  time_type t,
                  time_type t_min = std::numeric_limits<time_type>::lowest());

private:
    binning_kind policy_;

    // Interval in which event times can be aliased.
    time_type bin_interval_;

    // (Consider replacing this with a vector-backed store.)
    std::unordered_map<cell_gid_type, time_type> last_event_times_;

    util::optional<time_type> last_event_time(cell_gid_type id);

    void update_last_event_time(cell_gid_type id, time_type t);
};

} // namespace mc
} // namespace nest

