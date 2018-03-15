#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include <event_generator.hpp>

namespace arb {
namespace py {

struct regular_generator_desc {
    arb::cell_member_type target = {0, 0};
    float weight = 0;
    arb::time_type tstart = arb::max_time;
    arb::time_type tstop  = arb::max_time;
    arb::time_type dt = 100;

    regular_generator_desc() = default;

    arb::event_generator make_cpp() const {
        return arb::regular_generator(target, weight, tstart, dt, tstop);
    }
};

struct poisson_generator_desc {
    using poisson_generator = arb::poisson_generator<std::mt19937_64>;

    using rng_type = std::mt19937_64;
    using gen_type = arb::poisson_generator<rng_type>;

    arb::cell_member_type target = {0, 0};
    float weight = 0;
    // default empty time range
    arb::time_type tstart = arb::max_time;
    arb::time_type tstop  = arb::max_time;
    arb::time_type rate_per_ms = 0.01; // 10 Hz.
    rng_type::result_type seed = 0;

    poisson_generator_desc() = default;

    arb::event_generator make_cpp() const {
        return gen_type(target, weight, rng_type(seed), tstart, rate_per_ms, tstop);
    }
};

struct sequence_generator_desc {
    pybind11::list py_events;

    sequence_generator_desc() = default;

    arb::event_generator make_cpp() const {
        arb::pse_vector events;
        events.reserve(py_events.size());
        for (auto& e: py_events) {
            events.push_back(pybind11::cast<arb::postsynaptic_spike_event>(e));
        }

        return arb::vector_backed_generator(std::move(events));
    }
};

} // namespace py
} // namespace arb
