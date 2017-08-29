#pragma once

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <common_types.hpp>
#include <util/compat.hpp>
#include <util/debug.hpp>
#include <util/meta.hpp>

// Time schedules for probe–sampler associations.

namespace nest {
namespace mc {

// A schedule describes a sequence of time values used for sampling. Schedules
// are queried monotonically in time: if two method calls `events(t0, t1)` 
// and `events(t2, t3)` are made without an intervening call to `reset()`,
// then 0 ≤ _t0_ ≤ _t1_ ≤ _t2_ ≤ _t3_.

class schedule {
public:
    template <typename Impl>
    explicit schedule(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    template <typename Impl>
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

    std::vector<time_type> events(time_type t0, time_type t1) {
        return impl_->events(t0, t1);
    }

    void reset() { impl_->reset(); }

private:
    struct interface {
        virtual std::vector<time_type> events(time_type t0, time_type t1) = 0;
        virtual void reset() = 0;
        virtual std::unique_ptr<interface> clone() = 0;
        virtual ~interface() {}
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::vector<time_type> events(time_type t0, time_type t1) {
            return wrapped.events(t0, t1);
        }

        virtual void reset() {
            wrapped.reset();
        }

        virtual std::unique_ptr<interface> clone() {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        Impl wrapped;
    };
};


// Common schedules

// Schedule at k·dt for integral k≥0.
class regular_schedule_impl {
public:
    explicit regular_schedule_impl(time_type dt):
        dt_(dt), oodt_(1./dt) {};

    void reset() {}
    std::vector<time_type> events(time_type t0, time_type t1);

private:
    time_type dt_;
    time_type oodt_;
};

inline schedule regular_schedule(time_type dt) {
    return schedule(regular_schedule_impl(dt));
}

// Schedule at times given explicitly via a provided sorted sequence.
class explicit_schedule_impl {
public:
    template <typename Seq, typename = util::enable_if_sequence_t<Seq>>
    explicit explicit_schedule_impl(const Seq& seq):
        start_index_(0),
        times_(std::begin(seq), compat::end(seq))
    {
        EXPECTS(std::is_sorted(times_.begin(), times_.end()));
    }

    void reset() {
        start_index_ = 0;
    }

    std::vector<time_type> events(time_type t0, time_type t1);

private:
    std::ptrdiff_t start_index_;
    std::vector<time_type> times_;
};

template <typename Seq>
inline schedule explicit_schedule(const Seq& seq) {
    return schedule(explicit_schedule_impl(seq));
}

// Schedule at Poisson point process with rate 1/mean_dt,
// restricted to non-negative times.
template <typename RandomNumberEngine>
class poisson_schedule_impl {
public:
    poisson_schedule_impl(time_type tstart, time_type mean_dt, const RandomNumberEngine& rng):
        tstart_(tstart), exp_(1./mean_dt), rng_(rng), reset_state_(rng), next_(tstart)
    {
        EXPECTS(tstart_>=0);
        step();
    }

    void reset() {
        rng_ = reset_state_;
        next_ = tstart_;
        step();
    }

    std::vector<time_type> events(time_type t0, time_type t1) {
        std::vector<time_type> ts;

        while (next_<t0) {
            step();
        }

        while (next_<t1) {
            ts.push_back(next_);
            step();
        }

        return ts;
    }

private:
    void step() {
        next_ += exp_(rng_);
    }

    time_type tstart_;
    std::exponential_distribution<time_type> exp_;
    RandomNumberEngine rng_;
    RandomNumberEngine reset_state_;
    time_type next_;
};

template <typename RandomNumberEngine>
inline schedule poisson_schedule(time_type mean_dt, const RandomNumberEngine& rng) {
    return schedule(poisson_schedule_impl<RandomNumberEngine>(0., mean_dt, rng));
}

template <typename RandomNumberEngine>
inline schedule poisson_schedule(time_type tstart, time_type mean_dt, const RandomNumberEngine& rng) {
    return schedule(poisson_schedule_impl<RandomNumberEngine>(tstart, mean_dt, rng));
}

} // namespace mc
} // namespace nest
