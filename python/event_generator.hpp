#pragma once

#include <pybind11/pybind11.h>

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

    regular_schedule_shim(time_type t0, time_type dt, time_type t1):
        tstart(t0),
        dt(dt),
        tstop(t1)
    {}

    arb::schedule schedule() const {
        return arb::regular_schedule(tstart, dt, tstop);
    }
};

// A Python shim for arb::explicit_schedule.
// This is wrapped in pybind11, and users constructing a explicit_schedule in
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

struct poisson_schedule_shim {
    using rng_type = std::mt19937_64;

    // default empty time range
    arb::time_type tstart = 0;
    arb::time_type freq = 10; // 10 Hz.
    rng_type::result_type seed = 0;

    poisson_schedule_shim() = default;

    arb::schedule schedule() const {
        // convert frequency to kHz.
        return arb::poisson_schedule(tstart, freq/1000., rng_type(seed));
    }
};

struct event_generator {
    arb::cell_lid_type lid;
    double weight;
    arb::schedule time_seq;

    event_generator(arb::cell_lid_type lid, double weight, arb::schedule seq):
        lid(lid),
        weight(weight),
        time_seq(std::move(seq))
    {}
};

} // namespace pyarb

