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

    if (t1 > t0) {
        times_.reserve(1 + std::size_t((t1 - t0)*oodt_));

        long long n = t0*oodt_;
        time_type t;
        for (t = n*dt_; t < t0; t = (++n)*dt_) {}
        for (         ; t < t1; t = (++n)*dt_) times_.push_back(t);
    }

    return as_time_event_span(times_);
}

// Explicit schedule implementation.

time_event_span explicit_schedule_impl::events(time_type t0, time_type t1) {
    const auto& [tl, tr] = as_time_event_span(times_);
    auto lb = std::lower_bound(tl, tr, t0);
    auto ub = std::lower_bound(lb, tr, t1);
    return {lb, ub};
}

time_event_span poisson_schedule_impl::events(time_type t0, time_type t1) {
    // if we start after the maximal allowed time, we have nothing to do
    if (t0 >= tstop_) return {};

    // Restart rng cache
    index_ = 4;

    times_.clear();

    // restrict by maximal allowed time
    t1 = std::min(t1, tstop_);

    for (auto t = step(t0); t < t1; t = step(t)) times_.push_back(t);

    return as_time_event_span(times_);
}

time_type poisson_schedule_impl::step(time_type t) {
    using rng = r123::Threefry4x64;
    if (index_ >= 4) {
        auto r4 = rng{}(rng::ctr_type{(std::uint64_t)t}, rng::key_type{seed_});
        cache_ = r123::u01all<double>(r4);
        index_ = 0;
    }
    // convert uniform [0, 1) to exponential; adding up gives us Poisson.
    return t - std::log(1 - cache_[index_++])*oo_rate_;
}

} // namespace arb
