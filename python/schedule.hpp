#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/schedule.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>

namespace pyarb {

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.
struct regular_schedule_shim {
    using time_type = arb::time_type;
    using opt_time_type = arb::util::optional<time_type>;

    opt_time_type tstart = {};
    opt_time_type tstop = {};
    time_type dt = 0;

    regular_schedule_shim() = default;

    regular_schedule_shim(pybind11::object t0, time_type deltat, pybind11::object t1);

    // getter and setter (in order to assert when being set)
    void set_tstart(pybind11::object t);
    void set_tstop(pybind11::object t);
    void set_dt(time_type delta_t);

    opt_time_type get_tstart() const;
    time_type get_dt()         const;
    opt_time_type get_tstop()  const;

    arb::schedule schedule() const;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

// A Python shim for arb::explicit_schedule.
// This is wrapped in pybind11, and users constructing an explicit_schedule in
// Python are manipulating this type. This is converted to an
// arb::explicit_schedule when a C++ recipe is created from a Python recipe.
struct explicit_schedule_shim {
    std::vector<arb::time_type> times;

    explicit_schedule_shim() = default;
    explicit_schedule_shim(std::vector<arb::time_type> t);

    // getter and setter (in order to assert when being set)
    void set_times(std::vector<arb::time_type> t);
    std::vector<arb::time_type> get_times() const;

    arb::schedule schedule() const;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

// A Python shim for arb::poisson_schedule.
// This is wrapped in pybind11, and users constructing a poisson_schedule in
// Python are manipulating this type. This is converted to an
// arb::poisson_schedule when a C++ recipe is created from a Python recipe.
struct poisson_schedule_shim {
    using rng_type = std::mt19937_64;

    arb::time_type tstart = arb::terminal_time;
    arb::time_type freq = 10.;  // Hz
    rng_type::result_type seed = 0;

    poisson_schedule_shim() = default;
    poisson_schedule_shim(arb::time_type ts, arb::time_type f, rng_type::result_type s);

    void set_tstart(arb::time_type t);
    void set_freq(arb::time_type f);

    arb::time_type get_tstart() const;
    arb::time_type get_freq() const;

    arb::schedule schedule() const;

    std::vector<arb::time_type> events(arb::time_type t0, arb::time_type t1);
};

}
