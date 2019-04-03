#include <sstream>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

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

std::string schedule_explicit_string(const explicit_schedule_shim& e) {
  std::stringstream s;
  s << "<explicit_schedule: times " << e.py_times << " ms>";
  return s.str();
};

std::string schedule_regular_string(const regular_schedule_shim& r) {
  std::stringstream s;
  s << "<regular_schedule: tstart " << r.tstart << " ms"
    << ", dt " << r.dt << " ms"
    << ", tstop " << r.tstop << " ms" << ">";
  return s.str();
};

std::string schedule_poisson_string(const poisson_schedule_shim& p) {
  std::stringstream s;
  s << "<regular_schedule: tstart " << p.tstart << " ms"
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

// Common schedules
    // Regular schedule
    pybind11::class_<regular_schedule_shim> regular_schedule(m, "regular_schedule",
        "Describes a regular schedule with multiples of dt within the interval [tstart, tstop).");

    regular_schedule
        .def(pybind11::init<>(),
            "Construct an empty regular schedule with default arguments:\n"
            "  tstart: None.\n"
            "  dt:     0 ms.\n"
            "  tstop:  None.")
        .def(pybind11::init<arb::time_type>(),
            "dt"_a,
            "Construct a regular schedule starting at 0 ms and never ending with argument:\n"
            "  dt:     The interval between time points (in ms).\n")
        .def(pybind11::init<arb::time_type, arb::time_type, arb::time_type>(),
            "tstart"_a, "tstop"_a, "dt"_a,
            "Construct a regular schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence (in ms).\n"
            "  tstop:  No events delivered after this time (in ms).\n"
            "  dt:     The interval between time points (in ms).\n")
        .def_readwrite("tstart", &regular_schedule_shim::tstart,
            "The delivery time of the first event in the sequence (in ms).")
        .def_readwrite("tstop", &regular_schedule_shim::tstop,
            "No events delivered after this time (in ms).")
        .def_readwrite("dt", &regular_schedule_shim::dt,
            "The interval between time points (in ms).")
        .def("__str__", &schedule_regular_string)
        .def("__repr__",&schedule_regular_string);

    // Explicit schedule
    pybind11::class_<explicit_schedule_shim> explicit_schedule(m, "explicit_schedule",
        "Describes an explicit schedule at a predetermined (sorted) sequence of times.");

    explicit_schedule
        .def(pybind11::init<>(), "Construct an explicit schedule with an empty list of times.")
        .def(pybind11::init<pybind11::list>(),
            "times"_a,
            "Construct an explicit schedule with argument:\n"
            "  times: A list of times (in ms).")
        .def_readwrite("times", &explicit_schedule_shim::py_times,
            "A list of times (in ms).")
        .def("__str__", &schedule_explicit_string)
        .def("__repr__",&schedule_explicit_string);

    // Poisson schedule
    pybind11::class_<poisson_schedule_shim> poisson_schedule(m, "poisson_schedule",
        "Describes a schedule according to a Poisson process.");

    poisson_schedule
        .def(pybind11::init<>(),
            "Construct a Poisson schedule with default arguments:\n"
            "  tstart: 0 ms.\n"
            "  freq:   10 Hz.\n"
            "  seed:   Seed 0 for the random number generator.")
        .def(pybind11::init<arb::time_type, std::mt19937_64::result_type>(),
            "freq"_a, "seed"_a,
            "Construct a Poisson schedule with arguments:\n"
            "  freq:   The expected frequency (in Hz).\n"
            "  seed:   The seed for the random number generator.")
        .def(pybind11::init<arb::time_type, arb::time_type, std::mt19937_64::result_type>(),
            "tstart"_a, "freq"_a, "seed"_a,
            "Construct a Poisson schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence (in ms).\n"
            "  freq:   The expected frequency (in Hz).\n"
            "  seed:   The seed for the random number generator.")
        .def_readwrite("tstart", &poisson_schedule_shim::tstart,
            "The delivery time of the first event in the sequence (in ms).")
        .def_readwrite("freq", &poisson_schedule_shim::freq,
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
