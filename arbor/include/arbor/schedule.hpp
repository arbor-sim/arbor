#pragma once

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <random>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/extra_traits.hpp>
#include <arbor/export.hpp>
#include <arbor/serdes.hpp>
#include <arbor/units.hpp>

// Time schedules for probe–sampler associations.
namespace arb {

using engine_type = std::mt19937_64;
using seed_type = std::remove_cv_t<decltype(engine_type::default_seed)>;

constexpr static auto default_seed = engine_type::default_seed;

using time_event_span = std::pair<const time_type*, const time_type*>;

inline time_event_span as_time_event_span(const std::vector<time_type>& v) {
    return {v.data(), v.data() + v.size()};
}

// Type erased wrapper
// A schedule describes a sequence of time values used for sampling. Schedules
// are queried monotonically in time: if two method calls `events(t0, t1)`
// and `events(t2, t3)` are made without an intervening call to `reset()`,
// then 0 ≤ _t0_ ≤ _t1_ ≤ _t2_ ≤ _t3_.
struct ARB_ARBOR_API schedule {
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

    time_event_span events(time_type t0, time_type t1) { return impl_->events(t0, t1); }

    void reset() { impl_->reset(); }

    // Discard the next n events. Used in tests.
    auto discard(std::size_t n) { return impl_->discard(n); }

    template<typename K>
    friend ARB_ARBOR_API void serialize(serializer& s, const K& k, const schedule& v) { v.impl_->t_serialize(s, to_serdes_key(k)); }
    template<typename K>
    friend ARB_ARBOR_API void deserialize(serializer& s, const K& k, schedule& v) { v.impl_->t_deserialize(s, to_serdes_key(k
)); }

private:
    struct interface {
        virtual time_event_span events(time_type t0, time_type t1) = 0;
        virtual void reset() = 0;
        virtual void discard(std::size_t n) = 0;
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
        time_event_span events(time_type t0, time_type t1) override { return wrapped.events(t0, t1); }
        void reset() override { wrapped.reset(); }
        void discard(std::size_t n) override { return wrapped.discard(n); }
        iface_ptr clone() override { return std::make_unique<wrap<Impl>>(wrapped); }
        void t_serialize(serializer& s, const std::string& k) const override { wrapped.t_serialize(s, k); }
        void t_deserialize(serializer& s, const std::string& k) override { wrapped.t_deserialize(s, k); }

        Impl wrapped;
    };
};

// Constructors

/// Regular schedule with start `t0`, interval `dt`, and optional end `t1`.
schedule ARB_ARBOR_API regular_schedule(const units::quantity& t0,
                                        const units::quantity& dt,
                                        const units::quantity& t1 = std::numeric_limits<time_type>::max()*units::ms);

/// Regular schedule with interval `dt`.
schedule ARB_ARBOR_API regular_schedule(const units::quantity& dt);

schedule ARB_ARBOR_API explicit_schedule(const std::vector<units::quantity>& seq);
schedule ARB_ARBOR_API explicit_schedule_from_milliseconds(const std::vector<time_type>& seq);

schedule ARB_ARBOR_API poisson_schedule(const units::quantity& tstart,
                                        const units::quantity& rate,
                                        seed_type seed = default_seed,
                                        const units::quantity& tstop=terminal_time*units::ms);

schedule ARB_ARBOR_API poisson_schedule(const units::quantity& rate,
                                        seed_type seed = default_seed,
                                        const units::quantity& tstop=terminal_time*units::ms);

} // namespace arb
