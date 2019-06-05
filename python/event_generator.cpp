#include <stdexcept>
#include <sstream>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "conversion.hpp"
#include "error.hpp"
#include "event_generator.hpp"

namespace pyarb {

namespace {
auto is_nonneg = [](auto&& t){ return t>=0.; };
}

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

    regular_schedule_shim(pybind11::object t0, time_type deltat, pybind11::object t1) {
        set_tstart(t0);
        set_tstop(t1);
        set_dt(deltat);
    }

    // getter and setter (in order to assert when being set)
    void set_tstart(pybind11::object t) {
        tstart = py2optional<time_type>(t, "tstart must a non-negative number, or None", is_nonneg);
    };
    void set_tstop(pybind11::object t) {
        tstop = py2optional<time_type>(t, "tstop must a non-negative number, or None", is_nonneg);
    };
    void set_dt(time_type delta_t) {
        pyarb::assert_throw(is_nonneg(delta_t), "dt must be a non-negative number");
        dt = delta_t;
    };

    opt_time_type get_tstart() const { return tstart; }
    time_type get_dt()         const { return dt; }
    opt_time_type get_tstop()  const { return tstop; }

    arb::schedule schedule() const {
        return arb::regular_schedule(
                tstart.value_or(arb::terminal_time),
                dt,
                tstop.value_or(arb::terminal_time));
    }

};

// A Python shim for arb::explicit_schedule.
// This is wrapped in pybind11, and users constructing an explicit_schedule in
// Python are manipulating this type. This is converted to an
// arb::explicit_schedule when a C++ recipe is created from a Python recipe.

struct explicit_schedule_shim {
    using time_type = arb::time_type;

    std::vector<time_type> times;

    explicit_schedule_shim() = default;

    explicit_schedule_shim(std::vector<time_type> t) {
        set_times(t);
    }

    // getter and setter (in order to assert when being set)
    void set_times(std::vector<time_type> t) {
        times = std::move(t);
        // Sort the times in ascending order if necessary
        if (!std::is_sorted(times.begin(), times.end())) {
            std::sort(times.begin(), times.end());
        }
        // Assert that there are no negative times
        if (times.size()) {
            pyarb::assert_throw(is_nonneg(times[0]),
                    "explicit time schedule can not contain negative values");
        }
    };

    std::vector<time_type> get_times() const { return times; }

    arb::schedule schedule() const {
        return arb::explicit_schedule(times);
    }
};

// A Python shim for arb::poisson_schedule.
// This is wrapped in pybind11, and users constructing a poisson_schedule in
// Python are manipulating this type. This is converted to an
// arb::poisson_schedule when a C++ recipe is created from a Python recipe.

struct poisson_schedule_shim {
    using rng_type = std::mt19937_64;
    using time_type = arb::time_type;

    time_type tstart = arb::terminal_time;
    time_type freq = 10.;
    rng_type::result_type seed = 0;

    poisson_schedule_shim() = default;

    poisson_schedule_shim(time_type ts, time_type f, rng_type::result_type s) {
        set_tstart(ts);
        set_freq(f);
        seed = s;
    }

    void set_tstart(time_type t) {
        pyarb::assert_throw(is_nonneg(t), "tstart must be a non-negative number");
        tstart = t;
    };

    void set_freq(time_type f) {
        pyarb::assert_throw(is_nonneg(f), "frequency must be a non-negative number");
        freq = f;
    };

    const time_type get_tstart() const { return tstart; }
    const time_type get_freq() const { return freq; }

    arb::schedule schedule() const {
        // convert frequency to kHz.
        return arb::poisson_schedule(tstart, freq/1000., rng_type(seed));
    }
};

template <typename Sched>
event_generator_shim make_event_generator(
        arb::cell_member_type target,
        double weight,
        const Sched& sched)
{
    return event_generator_shim(target, weight, sched.schedule());
}

// Helper template for printing C++ optional types in Python.
// Prints either the value, or None if optional value is not set.
template <typename T>
std::string to_string(const arb::util::optional<T>& o, std::string unit) {
    if (!o) return "None";

    std::stringstream s;
    s << *o << " " << unit;
    return s.str();
}

std::string schedule_regular_string(const regular_schedule_shim& r) {
    std::stringstream s;
    s << "<regular_schedule: "
      << "tstart " << to_string(r.tstart, "ms") << ", "
      << "dt " << r.dt << " ms, "
      << "tstop " << to_string(r.tstop, "ms") << ">";
    return s.str();
};

std::string schedule_explicit_string(const explicit_schedule_shim& e) {
    std::stringstream s;
    s << "<explicit_schedule: times [";
    bool first = true;
    for (auto t: e.times) {
        if (!first) {
            s << " ";
        }
        s << t;
        first = false;
    }
    s << "] ms>";
    return s.str();
};

std::string schedule_poisson_string(const poisson_schedule_shim& p) {
    std::stringstream s;
    s << "<poisson_schedule: tstart " << p.tstart << " ms"
      << ", freq " << p.freq << " Hz"
      << ", seed " << p.seed << ">";
    return s.str();
};

void register_event_generators(pybind11::module& m) {
    using namespace pybind11::literals;
    using time_type = arb::time_type;

    // Regular schedule
    pybind11::class_<regular_schedule_shim> regular_schedule(m, "regular_schedule",
        "Describes a regular schedule with multiples of dt within the interval [tstart, tstop).");

    regular_schedule
        .def(pybind11::init<pybind11::object, time_type, pybind11::object>(),
            "tstart"_a = pybind11::none(), "dt"_a = 0., "tstop"_a = pybind11::none(),
            "Construct a regular schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence (in ms, default None).\n"
            "  dt:     The interval between time points (in ms, default 0).\n"
            "  tstop:  No events delivered after this time (in ms, default None).")
        .def_property("tstart", &regular_schedule_shim::get_tstart, &regular_schedule_shim::set_tstart,
            "The delivery time of the first event in the sequence (in ms).")
        .def_property("tstop", &regular_schedule_shim::get_tstop, &regular_schedule_shim::set_tstop,
            "No events delivered after this time (in ms).")
        .def_property("dt", &regular_schedule_shim::get_dt, &regular_schedule_shim::set_dt,
            "The interval between time points (in ms).")
        .def("__str__", &schedule_regular_string)
        .def("__repr__",&schedule_regular_string);

    // Explicit schedule
    pybind11::class_<explicit_schedule_shim> explicit_schedule(m, "explicit_schedule",
        "Describes an explicit schedule at a predetermined (sorted) sequence of times.");

    explicit_schedule
        .def(pybind11::init<>(),
            "Construct an empty explicit schedule.\n")
        .def(pybind11::init<std::vector<time_type>>(),
            "times"_a,
            "Construct an explicit schedule with argument:\n"
            "  times: A list of times (in ms, default []).")
        .def_property("times", &explicit_schedule_shim::get_times, &explicit_schedule_shim::set_times,
            "A list of times (in ms).")
        .def("__str__", &schedule_explicit_string)
        .def("__repr__",&schedule_explicit_string);

    // Poisson schedule
    pybind11::class_<poisson_schedule_shim> poisson_schedule(m, "poisson_schedule",
        "Describes a schedule according to a Poisson process.");

    poisson_schedule
        .def(pybind11::init<time_type, time_type, std::mt19937_64::result_type>(),
            "tstart"_a = 0., "freq"_a = 10., "seed"_a = 0,
            "Construct a Poisson schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence (in ms, default 0 ms).\n"
            "  freq:   The expected frequency (in Hz, default 10 Hz).\n"
            "  seed:   The seed for the random number generator (default 0).")
        .def_property("tstart", &poisson_schedule_shim::get_tstart, &poisson_schedule_shim::set_tstart,
            "The delivery time of the first event in the sequence (in ms).")
        .def_property("freq", &poisson_schedule_shim::get_freq, &poisson_schedule_shim::set_freq,
            "The expected frequency (in Hz).")
        .def_readwrite("seed", &poisson_schedule_shim::seed,
            "The seed for the random number generator.")
        .def("__str__", &schedule_poisson_string)
        .def("__repr__",&schedule_poisson_string);

// Event generator
    pybind11::class_<event_generator_shim> event_generator(m, "event_generator");

    event_generator
        .def(pybind11::init<>(
            [](arb::cell_member_type target, double weight, const regular_schedule_shim& sched){
                return make_event_generator(target, weight, sched);}),
            "target"_a, "weight"_a, "sched"_a,
            "Construct an event generator with arguments:\n"
            "  target: The target synapse (gid, local_id).\n"
            "  weight: The weight of events to deliver.\n"
            "  sched:  A regular schedule of the events.")
        .def(pybind11::init<>(
            [](arb::cell_member_type target, double weight, const explicit_schedule_shim& sched){
                return make_event_generator(target, weight, sched);}),
            "target"_a, "weight"_a, "sched"_a,
            "Construct an event generator with arguments:\n"
            "  target: The target synapse (gid, local_id).\n"
            "  weight: The weight of events to deliver.\n"
            "  sched:  An explicit schedule of the events.")
        .def(pybind11::init<>(
            [](arb::cell_member_type target, double weight, const poisson_schedule_shim& sched){
                return make_event_generator(target, weight, sched);}),
            "target"_a, "weight"_a, "sched"_a,
            "Construct an event generator with arguments:\n"
            "  target: The target synapse (gid, local_id).\n"
            "  weight: The weight of events to deliver.\n"
            "  sched:  A poisson schedule of the events.")
        .def_readwrite("target", &event_generator_shim::target,
             "The target synapse (gid, local_id).")
        .def_readwrite("weight", &event_generator_shim::weight,
             "The weight of events to deliver.")
        .def("__str__", [](const event_generator_shim&){return "<arbor.event_generator>";})
        .def("__repr__", [](const event_generator_shim&){return "<arbor.event_generator>";});
}

} // namespace pyarb
