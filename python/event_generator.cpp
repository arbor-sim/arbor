#include <stdexcept>
#include <sstream>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "exception.hpp"

namespace pyarb {

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.

auto is_nonneg = [](auto&& t){ return t>=0.; };

struct regular_schedule_shim {
    using time_type = arb::time_type;
    using opt_time_type = arb::util::optional<time_type>;
    static constexpr time_type default_tstart = arb::terminal_time;
    static constexpr time_type default_dt = 0.;
    static constexpr time_type default_tstop = arb::terminal_time;

    time_type tstart, dt, tstop;

    // getter and setter (in order to assert when being set)
    void set_tstart(const opt_time_type t0) {
        tstart = pyarb::assert_predicate(t0.value_or(default_tstart), is_nonneg, "tstart must be None, or a non-negative number.");
    };
    void set_tstop(const opt_time_type t1) {
        tstop = pyarb::assert_predicate(t1.value_or(default_tstop), is_nonneg, "tstop must be None, or a non-negative number.");
    };
    void set_dt(const time_type deltat) {
        dt = pyarb::assert_predicate(deltat, is_nonneg, "dt must be a non-negative number.");
    };
    const time_type get_tstart() const { return tstart; }
    const time_type get_tstop()  const { return tstop; }
    const time_type get_dt()     const { return dt; }

    regular_schedule_shim(opt_time_type t0, time_type deltat, opt_time_type t1):
        tstart(pyarb::assert_predicate(t0.value_or(default_tstart), is_nonneg, "tstart must be None, or a non-negative number.")),
        dt(    pyarb::assert_predicate(deltat,                      is_nonneg, "dt must be a non-negative number.")),
        tstop( pyarb::assert_predicate(t1.value_or(default_tstop),  is_nonneg, "tstop must be None, or a non-negative number."))
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

    std::list<time_type> times;

    explicit_schedule_shim(std::list<time_type> l) {
        for (auto& t: l) {
            times.push_back(pyarb::assert_predicate(t, is_nonneg, "time must be a non-negative number."));
        }

        // Sort the times in ascending order if necessary
        if (!std::is_sorted(times.begin(), times.end())) {
            times.sort();
        }

    }

    // getter and setter (in order to assert when being set)
    void set_times(const std::list<time_type> t0) {
        for (auto& t: t0) {
            times.push_back(pyarb::assert_predicate(t, is_nonneg, "time must be a non-negative number."));
        }
    };
    const std::list<time_type> get_times() const { return times; }

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

    time_type tstart = 0.;
    time_type freq = 10.;
    rng_type::result_type seed = 0;

    poisson_schedule_shim(time_type t0, time_type frequency, rng_type::result_type seeding):
        tstart(pyarb::assert_predicate(t0, is_nonneg, "tstart must be a non-negative number.")),
        freq(pyarb::assert_predicate(frequency, is_nonneg, "freq must be a non-negative number.")),
        seed(seeding)
    {}

    void set_tstart(const time_type t0) {
        tstart = pyarb::assert_predicate(t0, is_nonneg, "tstart must be a non-negative number.");
    };
    void set_freq(const time_type f) {
        freq = pyarb::assert_predicate(f, is_nonneg, "freq must be a non-negative number.");
    };
    const time_type get_tstart() const { return tstart; }
    const time_type get_freq() const { return freq; }

    arb::schedule schedule() const {
        // convert frequency to kHz.
        return arb::poisson_schedule(tstart, freq/1000., rng_type(seed));
    }
};

std::string schedule_regular_string(const regular_schedule_shim& r) {
    std::stringstream s;
    s << "<regular_schedule: tstart ";
    if (r.tstart == arb::terminal_time) {
        s << "None";
    }
    else
        s << r.tstart << " ms";
    s << ", dt " << r.dt << " ms"
    << ", tstop ";
    if (r.tstop == arb::terminal_time) {
        s << "None";
    }
    else
        s << r.tstop << " ms";
    s << ">";
    return s.str();
};

std::string schedule_explicit_string(const explicit_schedule_shim& e) {
    std::stringstream s;
    s << "<explicit_schedule: times [";
    std::list<arb::time_type>::const_iterator it = e.times.begin();
    for (auto t: e.times) {
        s << t;
        ++it;
        if (it != e.times.end()) s << " ";
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

template <typename Sched>
event_generator_shim make_event_generator(
        arb::cell_member_type target,
        double weight,
        const Sched& sched)
{
    return event_generator_shim(target, weight, sched.schedule());
}

void register_event_generators(pybind11::module& m) {
    using namespace pybind11::literals;
    using time_type = arb::time_type;
    using opt_time_type = arb::util::optional<time_type>;

// Common schedules
    // Regular schedule
    pybind11::class_<regular_schedule_shim> regular_schedule(m, "regular_schedule",
        "Describes a regular schedule with multiples of dt within the interval [tstart, tstop).");

    regular_schedule
        .def(pybind11::init<opt_time_type, time_type, opt_time_type>(),
            "tstart"_a = pybind11::none(), "dt"_a = 0., "tstop"_a = pybind11::none(),
            "Construct a regular schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence (in ms, default None [terminal time]).\n"
            "  dt:     The interval between time points (in ms, default 0).\n"
            "  tstop:  No events delivered after this time (in ms, default None [terminal time]).")
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
        .def(pybind11::init<std::list<time_type>>(),
            "times"_a = pybind11::list(),
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
