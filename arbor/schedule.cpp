#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

#include "util/cbrng.hpp"

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

time_event_span poisson_schedule_impl::events(time_type t0, time_type t1) {
    // if we start after the maximal allowed time, we have nothing to do
    if (t0 >= tstop_) return {};

    // restrict by maximal allowed time
    t1 = std::min(t1, tstop_);

    times_.clear();

    while (next_<t0) step();

    while (next_<t1) {
        times_.push_back(next_);
        step();
    }

    return as_time_event_span(times_);
}

void poisson_schedule_impl::step() {
    using rng = r123::Threefry4x64;
    if (index_ >= 4) {
        auto r4 = rng{}(rng::ctr_type{{0, (std::uint64_t)next_}}, rng::key_type{{seed_}});
        cache_ = r123::u01all<double>(r4);
        index_ = 0;
    }
    // convert uniform [0, 1) to exponential; adding up gives us Poisson.
    next_ += -std::log(1 - cache_[index_])*oo_rate_;
    index_++;
}

} // namespace arb
