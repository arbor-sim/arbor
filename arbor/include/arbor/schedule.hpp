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
#include <arbor/serdes.hpp>

// Time schedules for probe–sampler associations.
namespace arb {

using time_event_span = std::pair<const time_type*, const time_type*>;

inline time_event_span as_time_event_span(const std::vector<time_type>& v) {
    return {v.data(), v.data()+v.size()};
}

// Common schedules

// Default schedule is empty.

struct ARB_ARBOR_API empty_schedule {
    void reset() {}
    time_event_span events(time_type t0, time_type t1) {
        static time_type no_time;
        return {&no_time, &no_time};
    }
};

// Schedule at k·dt for integral k≥0 within the interval [t0, t1).
struct ARB_ARBOR_API regular_schedule_impl {
    explicit regular_schedule_impl(time_type t0, time_type dt, time_type t1):
        t0_(t0), t1_(t1), dt_(dt), oodt_(1./dt)
    {
        if (t0_<0) t0_ = 0;
    };

    void reset() {}
    time_event_span events(time_type t0, time_type t1);

    ARB_SERDES_ENABLE(regular_schedule_impl, t0_, t1_, dt_, oodt_);

private:
    time_type t0_, t1_, dt_;
    time_type oodt_;

    std::vector<time_type> times_;
};

// Schedule at times given explicitly via a provided sorted sequence.
struct ARB_ARBOR_API explicit_schedule_impl {
    explicit_schedule_impl(const explicit_schedule_impl&) = default;
    explicit_schedule_impl(explicit_schedule_impl&&) = default;

    template <typename Seq>
    explicit explicit_schedule_impl(const Seq& seq):
        start_index_(0)
    {
        using std::begin;
        using std::end;

        times_.assign(begin(seq), end(seq));
        arb_assert(std::is_sorted(times_.begin(), times_.end()));
    }

    void reset() {
        start_index_ = 0;
    }

    time_event_span events(time_type t0, time_type t1);

    ARB_SERDES_ENABLE(explicit_schedule_impl, start_index_, times_);

private:
    std::ptrdiff_t start_index_;
    std::vector<time_type> times_;
};

// Schedule at Poisson point process with rate 1/mean_dt,
// restricted to non-negative times.
template <typename RandomNumberEngine>
class poisson_schedule_impl {
public:
    poisson_schedule_impl(time_type tstart, time_type rate_kHz, const RandomNumberEngine& rng, time_type tstop):
        tstart_(tstart), exp_(rate_kHz), rng_(rng), reset_state_(rng), next_(tstart), tstop_(tstop)
    {
        arb_assert(tstart_>=0);
        arb_assert(tstart_ <= tstop_);
        step();
    }

    void reset() {
        rng_ = reset_state_;
        next_ = tstart_;
        step();
    }

    time_event_span events(time_type t0, time_type t1) {
        // if we start after the maximal allowed time, we have nothing to do
        if (t0 >= tstop_) return {};

        // restrict by maximal allowed time
        t1 = std::min(t1, tstop_);

        times_.clear();

        while (next_<t0) { step(); }

        while (next_<t1) {
            times_.push_back(next_);
            step();
        }

        return as_time_event_span(times_);
    }

    void step() { next_ += exp_(rng_); }

    time_type tstart_;
    std::exponential_distribution<time_type> exp_;
    RandomNumberEngine rng_;
    RandomNumberEngine reset_state_;
    time_type next_;
    std::vector<time_type> times_;
    time_type tstop_;
};

// Type erased wrapper
// A schedule describes a sequence of time values used for sampling. Schedules
// are queried monotonically in time: if two method calls `events(t0, t1)`
// and `events(t2, t3)` are made without an intervening call to `reset()`,
// then 0 ≤ _t0_ ≤ _t1_ ≤ _t2_ ≤ _t3_.

template<typename K>
void serialize(arb::serializer& s, const K& k, const arb::explicit_schedule_impl&);
template<typename K>
void deserialize(arb::serializer& s, const K& k, arb::explicit_schedule_impl&);

template<typename K>
void serialize(arb::serializer& s, const K& k, const arb::regular_schedule_impl&);
template<typename K>
void deserialize(arb::serializer& s, const K& k, arb::regular_schedule_impl&);

template <typename K>
ARB_ARBOR_API void serialize(::arb::serializer& ser,
                  const K& k,
                  const arb::empty_schedule& t) {
    ser.begin_write_map(::arb::to_serdes_key(k));
    ser.end_write_map();
}

template <typename K>
ARB_ARBOR_API void deserialize(::arb::serializer& ser,
                        const K& k,
                        arb::empty_schedule& t) {
    ser.begin_read_map(::arb::to_serdes_key(k));
    ser.end_read_map();
}

// These are custom to get the reset in.
template<typename K, typename R>
void serialize(::arb::serializer& ser, const K& k, const ::arb::poisson_schedule_impl<R>& t) {
    ser.begin_write_map(arb::to_serdes_key(k));
    ARB_SERDES_WRITE(tstart_);
    ARB_SERDES_WRITE(tstop_);
    ser.end_write_map();
}

template<typename K, typename R>
void deserialize(::arb::serializer& ser, const K& k, ::arb::poisson_schedule_impl<R>& t) {
    ser.begin_read_map(arb::to_serdes_key(k));
    ARB_SERDES_READ(tstart_);
    ARB_SERDES_READ(tstop_);
    ser.end_read_map();
    t.reset();
}

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

    void reset() { impl_->reset(); }

    template<typename K>
    friend ARB_ARBOR_API void serialize(serializer& s, const K& k, const schedule& v) { v.impl_->t_serialize(s, to_serdes_key(k)); }
    template<typename K>
    friend ARB_ARBOR_API void deserialize(serializer& s, const K& k, schedule& v) { v.impl_->t_deserialize(s, to_serdes_key(k
)); }

private:

    struct interface {
        virtual time_event_span events(time_type t0, time_type t1) = 0;
        virtual void reset() = 0;
        virtual std::unique_ptr<interface> clone() = 0;
        virtual ~interface() {}
        virtual void t_serialize(serializer&, const std::string&k) const = 0;
        virtual void t_deserialize(serializer&, const std::string&k) = 0;
    };

    using iface_ptr = std::unique_ptr<interface> ;

    iface_ptr impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}
        virtual time_event_span events(time_type t0, time_type t1) override { return wrapped.events(t0, t1); }
        virtual void reset() override { wrapped.reset(); }
        virtual iface_ptr clone() override { return std::make_unique<wrap<Impl>>(wrapped); }
        virtual void t_serialize(serializer& s, const std::string& k) const override { serialize(s, k, wrapped); }
        virtual void t_deserialize(serializer& s, const std::string& k) override { deserialize(s, k, wrapped); }

        Impl wrapped;
    };
};

// Constructors
inline schedule::schedule(): schedule(empty_schedule{}) {}

inline schedule regular_schedule(time_type t0,
                                 time_type dt,
                                 time_type t1 = std::numeric_limits<time_type>::max()) {
    return schedule(regular_schedule_impl(t0, dt, t1));
}

inline schedule regular_schedule(time_type dt) {
    return regular_schedule(0, dt);
}

template <typename Seq>
inline schedule explicit_schedule(const Seq& seq) {
    return schedule(explicit_schedule_impl(seq));
}

inline schedule explicit_schedule(const std::initializer_list<time_type>& seq) {
    return schedule(explicit_schedule_impl(seq));
}

template <typename RandomNumberEngine>
inline schedule poisson_schedule(time_type rate_kHz, const RandomNumberEngine& rng, time_type tstop=terminal_time) {
    return schedule(poisson_schedule_impl<RandomNumberEngine>(0., rate_kHz, rng, tstop));
}

template <typename RandomNumberEngine>
inline schedule poisson_schedule(time_type tstart, time_type rate_kHz, const RandomNumberEngine& rng, time_type tstop=terminal_time) {
    return schedule(poisson_schedule_impl<RandomNumberEngine>(tstart, rate_kHz, rng, tstop));
}

} // namespace arb
