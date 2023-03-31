#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/extra_traits.hpp>
#include <arbor/export.hpp>

// Time schedules for probe–sampler associations.

namespace arb {

using time_event_span = std::pair<const time_type*, const time_type*>;

inline time_event_span as_time_event_span(const std::vector<time_type>& v) {
    return {v.data(), v.data()+v.size()};
}

// A schedule describes a sequence of time values used for sampling. Schedules
// are queried monotonically in time: if two method calls `events(t0, t1)`
// and `events(t2, t3)` are made without an intervening call to `reset()`,
// then 0 ≤ _t0_ ≤ _t1_ ≤ _t2_ ≤ _t3_.

class schedule {
public:
    schedule();

    template <typename Impl, typename = std::enable_if_t<!std::is_same_v<util::remove_cvref_t<Impl>, schedule>>>
    explicit schedule(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    template <typename Impl, typename = std::enable_if_t<!std::is_same_v<util::remove_cvref_t<Impl>, schedule>>>
    explicit schedule(Impl&& impl):
        impl_(new wrap<Impl>(std::move(impl))) {}

    schedule(schedule&& other) = default;
    schedule& operator=(schedule&& other) = default;

    schedule(const schedule& other):
        impl_(other.impl_->clone()) {}

    schedule& operator=(const schedule& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    time_event_span events(time_type t0, time_type t1) {
        return impl_->events(t0, t1);
    }

private:
    struct interface {
        virtual time_event_span events(time_type t0, time_type t1) = 0;
        virtual std::unique_ptr<interface> clone() = 0;
        virtual ~interface() {}
    };

    using iface_ptr = std::unique_ptr<interface> ;

    iface_ptr impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}
        virtual time_event_span events(time_type t0, time_type t1) { return wrapped.events(t0, t1); }
        virtual iface_ptr clone() { return std::make_unique<wrap<Impl>>(wrapped); }

        Impl wrapped;
    };
};

// Default schedule is empty.

class empty_schedule {
public:
    time_event_span events(time_type t0, time_type t1) {
        static time_type no_time;
        return {&no_time, &no_time};
    }
};

inline schedule::schedule(): schedule(empty_schedule{}) {}

// Common schedules

// Schedule at k·dt for integral k≥0 within the interval [t0, t1).
class ARB_ARBOR_API regular_schedule_impl {
public:
    explicit regular_schedule_impl(time_type t0, time_type dt, time_type t1):
      t0_(std::max(0.0, t0)),
      t1_(t1),
      dt_(dt),
      oodt_(1./dt) {}

    time_event_span events(time_type t0, time_type t1);

private:
    time_type t0_, t1_, dt_;
    time_type oodt_;

    std::vector<time_type> times_;
};

inline schedule regular_schedule(time_type t0,
                                 time_type dt,
                                 time_type t1 = terminal_time) {
    return schedule(regular_schedule_impl(t0, dt, t1));
}

inline schedule regular_schedule(time_type dt) {
    return regular_schedule(0, dt);
}

// Schedule at times given explicitly via a provided sorted sequence.
struct ARB_ARBOR_API explicit_schedule_impl {
public:
    explicit_schedule_impl(const explicit_schedule_impl&) = default;
    explicit_schedule_impl(explicit_schedule_impl&&) = default;

    template <typename Seq>
    explicit explicit_schedule_impl(const Seq& seq) {
        times_.assign(std::begin(seq), std::end(seq));
        arb_assert(std::is_sorted(times_.begin(), times_.end()));
    }

    time_event_span events(time_type t0, time_type t1);

    std::vector<time_type> times_;
};

template <typename Seq>
inline schedule explicit_schedule(const Seq& seq) {
    return schedule(explicit_schedule_impl(seq));
}

inline schedule explicit_schedule(const std::initializer_list<time_type>& seq) {
    return schedule(explicit_schedule_impl(seq));
}

// Schedule at Poisson point process with rate 1/mean_dt,
// restricted to non-negative times.

struct ARB_ARBOR_API poisson_schedule_impl {
    poisson_schedule_impl(time_type tstart, time_type rate_kHz, time_type tstop, cell_gid_type seed):
        tstart_(tstart), tstop_(tstop), oo_rate_(1.0/rate_kHz), seed_{seed} {
        arb_assert(tstart_>=0);
        arb_assert(tstart_ <= tstop_);
    }
    time_event_span events(time_type t0, time_type t1);

private:
    time_type step(time_type);

    time_type tstart_;
    time_type tstop_;

    time_type oo_rate_;
    std::size_t index_ = 4;
    std::uint64_t seed_;
    std::array<time_type, 4> cache_;
    std::vector<time_type> times_;
};

inline schedule poisson_schedule(time_type rate_kHz, cell_gid_type seed, time_type tstop=terminal_time) {
    return schedule(poisson_schedule_impl(0., rate_kHz, tstop, seed));
}

inline schedule poisson_schedule(time_type tstart, time_type rate_kHz, cell_gid_type seed, time_type tstop=terminal_time) {
    return schedule(poisson_schedule_impl(tstart, rate_kHz, tstop, seed));
}

} // namespace arb
