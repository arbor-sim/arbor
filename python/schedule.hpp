#pragma once

#include <optional>
#include <random>
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
    schedule_shim_base(const schedule_shim_base&) = delete;
    schedule_shim_base& operator=(schedule_shim_base&) = delete;
    virtual ~schedule_shim_base() {}

    virtual arb::schedule schedule() const = 0;
};

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.
struct regular_schedule_shim: schedule_shim_base {
    using time_type = arb::time_type;
    using opt_time_type = std::optional<time_type>;

    time_type tstart = {};
    time_type dt = 0;
    opt_time_type tstop = {};

    regular_schedule_shim(time_type t0, time_type delta_t, pybind11::object t1);
    explicit regular_schedule_shim(time_type delta_t);

    // getter and setter (in order to assert when being set)
    void set_tstart(time_type t);
    void set_dt(time_type delta_t);
    void set_tstop(pybind11::object t);

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
    explicit_schedule_shim(std::vector<arb::time_type> t);

    // getter and setter (in order to assert when being set)
    void set_times(std::vector<arb::time_type> t);
    std::vector<arb::time_type> get_times() const;

    arb::schedule schedule() const override;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

// A Python shim for arb::poisson_schedule.
// This is wrapped in pybind11, and users constructing a poisson_schedule in
// Python are manipulating this type. This is converted to an
// arb::poisson_schedule when a C++ recipe is created from a Python recipe.
struct poisson_schedule_shim: schedule_shim_base {
    using rng_type = std::mt19937_64;
    using opt_time_type = std::optional<arb::time_type>;

    arb::time_type tstart; // ms
    arb::time_type freq; // kHz
    opt_time_type  tstop; // ms
    rng_type::result_type seed;

    poisson_schedule_shim(arb::time_type ts, arb::time_type f, rng_type::result_type s, pybind11::object tstop);
    poisson_schedule_shim(arb::time_type f);

    void set_tstart(arb::time_type t);
    void set_freq(arb::time_type f);
    void set_tstop(pybind11::object t);

    arb::time_type get_tstart() const;
    arb::time_type get_freq() const;
    opt_time_type get_tstop() const;

    arb::schedule schedule() const override;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

}
