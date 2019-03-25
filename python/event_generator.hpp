#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

namespace pyarb {

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.

struct regular_schedule_shim {
    using time_type = arb::time_type;

    time_type tstart = arb::terminal_time;
    time_type dt = 0;
    time_type tstop = arb::terminal_time;

    regular_schedule_shim() = default;

    regular_schedule_shim(time_type t0, time_type deltat, time_type t1):
        tstart(t0),
        dt(deltat),
        tstop(t1)
    {}

    regular_schedule_shim(time_type deltat):
        tstart(0),
        dt(deltat)
    {}

    arb::schedule schedule() const {
        return arb::regular_schedule(tstart, dt, tstop);
    }
};

// A Python shim for arb::explicit_schedule.
// This is wrapped in pybind11, and users constructing an explicit_schedule in
// Python are manipulating this type. This is converted to an
// arb::explicit_schedule when a C++ recipe is created from a Python recipe.

struct explicit_schedule_shim {
    using time_type = arb::time_type;
    pybind11::list py_times;

    explicit_schedule_shim() = default;

    arb::schedule schedule() const {
        std::vector<time_type> times;

        times.reserve(py_times.size());
        for (auto& t: py_times) {
            times.push_back(pybind11::cast<time_type>(t));
        }

        // Sort the times in ascending order if necessary
        if (!std::is_sorted(times.begin(), times.end())) {
            std::sort(times.begin(), times.end());
        }

        return arb::explicit_schedule(times);
    }
};

// A Python shim for arb::poisson_schedule.
// This is wrapped in pybind11, and users constructing a poisson_schedule in
// Python are manipulating this type. This is converted to an
// arb::poisson_schedule when a C++ recipe is created from a Python recipe.

struct poisson_schedule_shim {
    using rng_type = std::mt19937_64;

    // default empty time range
    arb::time_type tstart = 0;
    arb::time_type freq = 10; // 10 Hz.
    rng_type::result_type seed = 0;

    poisson_schedule_shim() = default;

    poisson_schedule_shim(arb::time_type frequency, rng_type::result_type seeding):
        tstart(0),
        freq(frequency),
        seed(seeding)
    {}

    poisson_schedule_shim(arb::time_type t0, arb::time_type frequency, rng_type::result_type seeding):
        tstart(t0),
        freq(frequency),
        seed(seeding)
    {}

    arb::schedule schedule() const {
        // convert frequency to kHz.
        return arb::poisson_schedule(tstart, freq/1000., rng_type(seed));
    }
};

struct event_generator_shim {
    arb::cell_member_type target;
    double weight;
    arb::schedule time_sched;

    event_generator_shim(arb::cell_member_type cell, double event_weight, arb::schedule sched):
        target(cell),
        weight(event_weight),
        time_sched(std::move(sched))
    {}
};

} // namespace pyarb
