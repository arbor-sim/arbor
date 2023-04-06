#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

// Implementations for specific schedules.

namespace arb {

// Regular schedule implementation.

time_event_span regular_schedule_impl::events(time_type t0, time_type t1) {
    times_.clear();

    t0 = std::max(t0, t0_);
    t1 = std::min(t1, t1_);

    if (t1>t0) {
        times_.reserve(1+std::size_t((t1-t0)*oodt_));

        long long n = t0*oodt_;
        time_type t = n*dt_;

        while (t<t0) {
            t = (++n)*dt_;
        }

        while (t<t1) {
            times_.push_back(t);
            t = (++n)*dt_;
        }
    }

    return as_time_event_span(times_);
}

// Explicit schedule implementation.

time_event_span explicit_schedule_impl::events(time_type t0, time_type t1) {
    time_event_span view = as_time_event_span(times_);

    const time_type* lb = std::lower_bound(view.first+start_index_, view.second, t0);
    const time_type* ub = std::lower_bound(lb, view.second, t1);

    start_index_ = ub-view.first;
    return {lb, ub};
}

} // namespace arb
