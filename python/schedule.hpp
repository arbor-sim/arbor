#pragma once

#include <optional>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/schedule.hpp>
#include <arbor/common_types.hpp>

namespace pyarb {

// Schedule shim base class provides virtual interface for conversion
// to an arb::schedule object.
struct schedule_shim_base {
    schedule_shim_base() = default;
    virtual ~schedule_shim_base() {}

    virtual arb::schedule schedule() const = 0;
};

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.
struct regular_schedule_shim: schedule_shim_base {
    using time_type = arb::units::quantity;
    using opt_time_type = std::optional<time_type>;

    time_type tstart;
    time_type dt = 0*arb::units::ms;
    opt_time_type tstop;

    regular_schedule_shim(const time_type& t0,
                          const time_type& delta_t,
                          opt_time_type t1);

    explicit regular_schedule_shim(const time_type& delta_t);

    // getter and setter (in order to assert when being set)
    void set_tstart(const time_type& t);
    void set_dt(const time_type& delta_t);
    void set_tstop(opt_time_type t);

    time_type get_tstart() const;
    time_type get_dt() const;
    opt_time_type get_tstop() const;

    arb::schedule schedule() const override;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

// A Python shim for arb::explicit_schedule.
// This is wrapped in pybind11, and users constructing an explicit_schedule in
// Python are manipulating this type. This is converted to an
// arb::explicit_schedule when a C++ recipe is created from a Python recipe.
struct explicit_schedule_shim: schedule_shim_base {
    std::vector<arb::time_type> times;

    explicit_schedule_shim() = default;
    explicit_schedule_shim(const std::vector<arb::units::quantity>& t);

    // getter and setter (in order to assert when being set)
    void set_times_ms(std::vector<arb::time_type> t);
    std::vector<arb::time_type> get_times_ms() const;

    arb::schedule schedule() const override;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

// A Python shim for arb::poisson_schedule.
// This is wrapped in pybind11, and users constructing a poisson_schedule in
// Python are manipulating this type. This is converted to an
// arb::poisson_schedule when a C++ recipe is created from a Python recipe.
struct poisson_schedule_shim: schedule_shim_base {
    arb::units::quantity tstart; // ms
    arb::units::quantity freq;   // kHz
    arb::units::quantity tstop; // ms
    arb::seed_type seed = arb::default_seed;

    poisson_schedule_shim(const poisson_schedule_shim&) = default;
    poisson_schedule_shim(poisson_schedule_shim&&) = default;

    poisson_schedule_shim() = default;
    ~poisson_schedule_shim() = default;

    poisson_schedule_shim(const arb::units::quantity& ts,
                          const arb::units::quantity& f,
                          arb::seed_type s,
                          const arb::units::quantity& tstop);

    void set_tstart(const arb::units::quantity& t);
    void set_freq(const arb::units::quantity& f);
    void set_tstop(const arb::units::quantity& f);

    const auto& get_tstop() const { return tstop; }
    const auto& get_tstart() const { return tstart; }
    const auto& get_freq() const { return freq; }

    // TODO(TH) this should be symmetrical...
    std::vector<arb::time_type> events(const arb::units::quantity& t0, const arb::units::quantity& t1);

    arb::schedule schedule() const;

};

}
