#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include <event_generator.hpp>

namespace arb {
namespace py {

// Trampoline class for user-defined event_generators in Pythonk.
struct event_generator: public arb::event_generator {
    using arb::event_generator::event_generator;

    arb::postsynaptic_spike_event next() override {
        PYBIND11_OVERLOAD_PURE(arb::postsynaptic_spike_event, arb::event_generator, next,);
    }
    void pop() override {
        PYBIND11_OVERLOAD_PURE(void, arb::event_generator, pop,);
    }
    void reset() override {
        PYBIND11_OVERLOAD_PURE(void, arb::event_generator, reset,);
    }
    void advance(arb::time_type t) override {
        PYBIND11_OVERLOAD_PURE(void, arb::event_generator, advance,);
    }
};

using poisson_generator = arb::poisson_generator<std::mt19937_64>;

/*
template <typename T>
arb::event_generator_ptr
make_cpp_event_generator(const T& py_gen, arb::cell_gid_type gid) {
    throw std::runtime_error("This can't be turned into an event generator.");
}

struct regular_generator_desc {
    arb::cell_member_type target = {0, 0};
    float weight = 0;
    arb::time_type tstart = 0;
    arb::time_type tstop = arb::max_time;
    arb::time_type dt = 100;

    regular_generator_desc() = default;

    arb::event_generator_ptr make_cpp() {
        return arb::make_event_generator<arb::regular_generator>(
                target, weight, tstart, dt, tstop);
    }
};

struct poisson_generator_desc {
    using rng_type = std::mt19937_64;
    using gen_type = arb::poisson_generator<rng_type>;

    arb::cell_member_type target = {0, 0};
    float weight = 0;
    arb::time_type tstart = 0;
    arb::time_type tstop = arb::max_time;
    arb::time_type rate_per_ms = 0.01; // 10 Hz.
    rng_type::result_type seed = 0;

    poisson_generator_desc() = default;

    arb::event_generator_ptr make_cpp() {
        return arb::make_event_generator<gen_type>(
            target, weight, rng_type(seed), tstart, rate_per_ms, tstop);
    }
};

struct sequence_generator_desc {
    pybind11::list py_events;

    arb::event_generator_ptr make_cpp() {
        arb::pse_vector events;
        events.reserve(py_events.size());
        for (auto& e: py_events) {
            events.push_back(pybind11::cast<arb::postsynaptic_spike_event>(e));
        }

        return arb::make_event_generator<
                arb::seq_generator<arb::pse_vector>>(std::move(events));
    }
};
*/

} // namespace py
} // namespace arb
