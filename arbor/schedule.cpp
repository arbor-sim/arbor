#include <algorithm>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

#include <iostream>

namespace arb {

// Schedule at Poisson point process with rate 1/mean_dt,
// restricted to non-negative times.
struct poisson_schedule_impl {
    poisson_schedule_impl(time_type tstart, time_type rate_kHz, seed_type seed, time_type tstop):
        tstart_(tstart), rate_(rate_kHz), exp_(rate_kHz), rng_(seed), seed_(seed), next_(tstart), tstop_(tstop) {
        if (!std::isfinite(tstart_))  throw std::domain_error("Poisson schedule: start must be finite and in [ms]");
        if (!std::isfinite(tstop_))   throw std::domain_error("Poisson schedule: stop must be finite and in [ms]");
        if (!std::isfinite(rate_kHz)) throw std::domain_error("Poisson schedule: rate must be finite and in [kHz]");
        if (!std::isfinite(tstart_) || tstart_ < 0) throw std::domain_error("Poisson schedule: start must be >= 0 and finite.");
        if (!std::isfinite(tstop_)  || tstop_ < tstart_) throw std::domain_error("Poisson schedule: stop must be >= start and finite.");
        step();
    }

    void reset() {
        rng_ = engine_type{seed_};
        if (discard_ > 0) rng_.discard(discard_);
        exp_ = std::exponential_distribution<time_type>{rate_};
        next_ = tstart_;
        step();
    }

    void discard(std::size_t n) { discard_ = n; reset(); }

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

    template<typename K>
    void t_serialize(::arb::serializer& ser, const K& k) const {
        const auto& t = *this;
        ser.begin_write_map(arb::to_serdes_key(k));
        ARB_SERDES_WRITE(tstart_);
        ARB_SERDES_WRITE(tstop_);
        ser.end_write_map();
    }

    template<typename K>
    void t_deserialize(::arb::serializer& ser, const K& k) {
        auto& t = *this;
        ser.begin_read_map(arb::to_serdes_key(k));
        ARB_SERDES_READ(tstart_);
        ARB_SERDES_READ(tstop_);
        ser.end_read_map();
        t.reset();
    }

    time_type tstart_;
    time_type rate_;
    std::exponential_distribution<time_type> exp_;
    engine_type rng_;
    seed_type seed_;
    time_type next_;
    std::vector<time_type> times_;
    time_type tstop_;
    std::size_t discard_ = 0;
};

schedule poisson_schedule(const units::quantity& tstart,
                          const units::quantity& rate,
                          seed_type seed,
                          const units::quantity& tstop) {
    return schedule(poisson_schedule_impl(tstart.value_as(units::ms),
                                          rate.value_as(units::kHz),
                                          seed,
                                          tstop.value_as(units::ms)));
}

schedule poisson_schedule(const units::quantity& rate,
                          seed_type seed,
                          const units::quantity& tstop) {
    return poisson_schedule(0.*units::ms, rate, seed, tstop);
}


struct empty_schedule {
    void reset() {}
    time_event_span events(time_type t0, time_type t1) {
        static time_type no_time;
        return {&no_time, &no_time};
    }


    void t_serialize(::arb::serializer& ser,
                  const std::string& k) const {
        ser.begin_write_map(::arb::to_serdes_key(k));
        ser.end_write_map();
    }

    void t_deserialize(::arb::serializer& ser,
                     const std::string& k) {
        ser.begin_read_map(::arb::to_serdes_key(k));
        ser.end_read_map();
    }

    void discard(std::size_t) {}
};

schedule::schedule(): schedule(empty_schedule{}) {}

// Schedule at k·dt for integral k≥0 within the interval [t0, t1).
struct ARB_ARBOR_API regular_schedule_impl {
    explicit regular_schedule_impl(time_type t0, time_type dt, time_type t1):
        t0_(t0), t1_(t1), dt_(dt), oodt_(1./dt) {
        if (!std::isfinite(t0_)) throw std::domain_error("Regular schedule: start must be finite and in [ms]");
        if (!std::isfinite(t1_)) throw std::domain_error("Regular schedule: stop must be finite and in [ms]");
        if (!std::isfinite(dt_)) throw std::domain_error("Regular schedule: step must be finite and in [ms]");
        if (dt_ <= 0)  throw std::domain_error("regular schedule: dt must be > 0 and finite.");
        if (t0_ < 0)   throw std::domain_error("regular schedule: start must be >= 0 and finite.");
        if (t1_ < t0_) throw std::domain_error("regular schedule: stop must be >= start and finite.");
    };

    void reset() {}

    ARB_SERDES_ENABLE(regular_schedule_impl, t0_, t1_, dt_, oodt_);

    template<typename K>
    void t_serialize(::arb::serializer& ser, const K& k) const {
        const auto& t = *this;
        ser.begin_write_map(arb::to_serdes_key(k));
        ARB_SERDES_WRITE(t0_);
        ARB_SERDES_WRITE(t1_);
        ARB_SERDES_WRITE(dt_);
        ser.end_write_map();
    }

    template<typename K>
    void t_deserialize(::arb::serializer& ser, const K& k) {
        auto& t = *this;
        ser.begin_read_map(arb::to_serdes_key(k));
        ARB_SERDES_READ(t0_);
        ARB_SERDES_READ(t1_);
        ARB_SERDES_READ(dt_);
        oodt_ = 1.0/dt_;
        ser.end_read_map();
    }

    time_event_span events(time_type t0, time_type t1) {
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

    void discard(std::size_t) {}

    time_type t0_, t1_, dt_;
    time_type oodt_;

    std::vector<time_type> times_;
};

schedule regular_schedule(const units::quantity& t0,
                          const units::quantity& dt,
                          const units::quantity& t1) {
    return schedule(regular_schedule_impl(t0.value_as(units::ms),
                                          dt.value_as(units::ms),
                                          t1.value_as(units::ms)));
}

schedule regular_schedule(const units::quantity& dt) { return regular_schedule(0*units::ms, dt); }

// Schedule at times given explicitly via a provided sorted sequence.
struct explicit_schedule_impl {
    explicit_schedule_impl(const explicit_schedule_impl&) = default;
    explicit_schedule_impl(explicit_schedule_impl&&) = default;

    explicit explicit_schedule_impl(std::vector<time_type> seq):
        start_index_(0), times_(std::move(seq)) {
        time_type last = -1;
        for (auto t: times_) {
            if (!std::isfinite(t)) throw std::domain_error("explicit schedule: times must be finite and in [ms]");
            if (t < 0)             throw std::domain_error("explicit schedule: times must be >= 0 and finite.");
            if (t < last)          throw std::domain_error("explicit schedule: times must be sorted.");
            last = t;
        }
    }

    void reset() { start_index_ = 0; }

    time_event_span events(time_type t0, time_type t1) {
        time_event_span view = as_time_event_span(times_);

        const time_type* lb = std::lower_bound(view.first+start_index_, view.second, t0);
        const time_type* ub = std::lower_bound(lb, view.second, t1);

        start_index_ = ub-view.first;
        return {lb, ub};
    }

    template<typename K>
    void t_serialize(::arb::serializer& ser, const K& k) const {
        const auto& t = *this;
        ser.begin_write_map(arb::to_serdes_key(k));
        ARB_SERDES_WRITE(start_index_);
        ARB_SERDES_WRITE(times_);
        ser.end_write_map();
    }

    template<typename K>
    void t_deserialize(::arb::serializer& ser, const K& k) {
        auto& t = *this;
        ser.begin_read_map(arb::to_serdes_key(k));
        ARB_SERDES_READ(start_index_);
        ARB_SERDES_READ(times_);
        ser.end_read_map();
    }

    void discard(std::size_t) {}

    std::ptrdiff_t start_index_;
    std::vector<time_type> times_;
};

schedule explicit_schedule(const std::vector<units::quantity>& seq) {
    std::vector<time_type> res;
    for (const auto& t: seq) res.push_back(t.value_as(units::ms));
    return schedule(explicit_schedule_impl(res));
}

schedule explicit_schedule_from_milliseconds(const std::vector<time_type>& seq) {
    return schedule(explicit_schedule_impl(seq));
}


} // namespace arb
