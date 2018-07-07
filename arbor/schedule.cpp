#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

// Implementations for specific schedules.

namespace arb {

// Regular schedule implementation.

std::vector<time_type> regular_schedule_impl::events(time_type t0, time_type t1) {
    std::vector<time_type> ts;

    t0 = t0<0? 0: t0;
    if (t1>t0) {
        ts.reserve(1+std::size_t((t1-t0)*oodt_));

        long long n = t0*oodt_;
        time_type t = n*dt_;

        while (t<t0) {
            t = (++n)*dt_;
        }

        while (t<t1) {
            ts.push_back(t);
            t = (++n)*dt_;
        }
    }

    return ts;
}

// Explicit schedule implementation.

std::vector<time_type> explicit_schedule_impl::events(time_type t0, time_type t1) {
    auto lb = std::lower_bound(times_.begin()+start_index_, times_.end(), t0);
    auto ub = std::lower_bound(times_.begin()+start_index_, times_.end(), t1);

    start_index_ = std::distance(times_.begin(), ub);
    return std::vector<time_type>(lb, ub);
}

} // namespace arb
