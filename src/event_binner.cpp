#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include <common_types.hpp>
#include <event_binner.hpp>
#include <spike.hpp>
#include <util/optional.hpp>

namespace arb {

void event_binner::reset() {
    last_event_times_.clear();
}

time_type event_binner::bin(cell_gid_type id, time_type t, time_type t_min) {
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
        if (auto last_t = last_event_time(id)) {
            if (t-*last_t<bin_interval_) {
                t_binned = *last_t;
            }
        }
        update_last_event_time(id, t_binned);
        break;
    default:
        throw std::logic_error("unrecognized binning policy");
    }

    return std::max(t_binned, t_min);
}

util::optional<time_type>
event_binner::last_event_time(cell_gid_type id) {
    auto it = last_event_times_.find(id);
    return it==last_event_times_.end()? util::nothing: util::just(it->second);
}

void event_binner::update_last_event_time(cell_gid_type id, time_type t) {
    last_event_times_[id] = t;
}

} // namespace arb

