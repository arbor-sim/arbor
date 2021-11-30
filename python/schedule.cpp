#include <arbor/schedule.hpp>
#include <arbor/common_types.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conversion.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"

namespace py = pybind11;

namespace pyarb {

std::ostream& operator<<(std::ostream& o, const regular_schedule_shim& x) {
    return o << "<arbor.regular_schedule: tstart "
             << util::to_string(x.tstart) << " ms, dt "
             << x.dt << " ms, tstop "
             << util::to_string(x.tstop) << " ms>";
}

std::ostream& operator<<(std::ostream& o, const explicit_schedule_shim& e) {
    o << "<arbor.explicit_schedule: times [";
    return o << util::csv(e.times) << "] ms>";
};

std::ostream& operator<<(std::ostream& o, const poisson_schedule_shim& p) {
    return o << "<arbor.poisson_schedule: tstart " << p.tstart << " ms"
             << ", tstop " << util::to_string(p.tstop) << " ms"
             << ", freq " << p.freq << " kHz"
             << ", seed " << p.seed << ">";
};

static std::vector<arb::time_type> as_vector(std::pair<const arb::time_type*, const arb::time_type*> ts) {
    return std::vector<arb::time_type>(ts.first, ts.second);
}

//
// regular_schedule shim
//

regular_schedule_shim::regular_schedule_shim(arb::time_type t0, arb::time_type delta_t, py::object t1) {
    set_tstart(t0);
    set_dt(delta_t);
    set_tstop(t1);
}

regular_schedule_shim::regular_schedule_shim(arb::time_type delta_t) {
    set_tstart(0.);
    set_dt(delta_t);
}

void regular_schedule_shim::set_tstart(arb::time_type t) {
    pyarb::assert_throw(is_nonneg()(t), "tstart must be a non-negative number");
    tstart = t;
};

void regular_schedule_shim::set_tstop(py::object t) {
    tstop = py2optional<time_type>(
            t, "tstop must be a non-negative number, or None", is_nonneg());
};

void regular_schedule_shim::set_dt(arb::time_type delta_t) {
    pyarb::assert_throw(is_positive()(delta_t), "dt must be a positive number");
    dt = delta_t;
};

regular_schedule_shim::time_type regular_schedule_shim::get_tstart() const {
    return tstart;
}

regular_schedule_shim::time_type regular_schedule_shim::get_dt() const {
    return dt;
}

regular_schedule_shim::opt_time_type regular_schedule_shim::get_tstop() const {
    return tstop;
}

arb::schedule regular_schedule_shim::schedule() const {
    return arb::regular_schedule(
            tstart,
            dt,
            tstop.value_or(arb::terminal_time));
}

std::vector<arb::time_type> regular_schedule_shim::events(arb::time_type t0, arb::time_type t1) {
    pyarb::assert_throw(is_nonneg()(t0), "t0 must be a non-negative number");
    pyarb::assert_throw(is_nonneg()(t1), "t1 must be a non-negative number");

    arb::schedule sched = regular_schedule_shim::schedule();

    return as_vector(sched.events(t0, t1));
}

//
// explicit_schedule shim
//

//struct explicit_schedule_shim {
explicit_schedule_shim::explicit_schedule_shim(std::vector<arb::time_type> t) {
    set_times(t);
}

// getter and setter (in order to assert when being set)
void explicit_schedule_shim::set_times(std::vector<arb::time_type> t) {
    times = std::move(t);

    // Sort the times in ascending order if necessary
    if (!std::is_sorted(times.begin(), times.end())) {
        std::sort(times.begin(), times.end());
    }

    // Assert that there are no negative times
    if (times.size()) {
        pyarb::assert_throw(is_nonneg()(times[0]),
                "explicit time schedule cannot contain negative values");
    }
};

std::vector<arb::time_type> explicit_schedule_shim::get_times() const {
    return times;
}

arb::schedule explicit_schedule_shim::schedule() const {
    return arb::explicit_schedule(times);
}

std::vector<arb::time_type> explicit_schedule_shim::events(arb::time_type t0, arb::time_type t1) {
    pyarb::assert_throw(is_nonneg()(t0), "t0 must be a non-negative number");
    pyarb::assert_throw(is_nonneg()(t1), "t1 must be a non-negative number");

    arb::schedule sched = explicit_schedule_shim::schedule();

    return as_vector(sched.events(t0, t1));
}

//
// poisson_schedule shim
//

poisson_schedule_shim::poisson_schedule_shim(
        arb::time_type ts,
        arb::time_type f,
        rng_type::result_type s,
        py::object tstop)
{
    set_tstart(ts);
    set_freq(f);
    seed = s;
    set_tstop(tstop);
}

poisson_schedule_shim::poisson_schedule_shim(arb::time_type f) {
    set_tstart(0.);
    set_freq(f);
    seed = 0;
}

void poisson_schedule_shim::set_tstart(arb::time_type t) {
    pyarb::assert_throw(is_nonneg()(t), "tstart must be a non-negative number");
    tstart = t;
};

void poisson_schedule_shim::set_freq(arb::time_type f) {
    pyarb::assert_throw(is_nonneg()(f), "frequency must be a non-negative number");
    freq = f;
};

void poisson_schedule_shim::set_tstop(py::object t) {
    tstop = py2optional<arb::time_type>(
            t, "tstop must be a non-negative number, or None", is_nonneg());
};

arb::time_type poisson_schedule_shim::get_tstart() const {
    return tstart;
}

arb::time_type poisson_schedule_shim::get_freq() const {
    return freq;
}

poisson_schedule_shim::opt_time_type poisson_schedule_shim::get_tstop() const {
    return tstop;
}

arb::schedule poisson_schedule_shim::schedule() const {
    return arb::poisson_schedule(tstart, freq, rng_type(seed), tstop.value_or(arb::terminal_time));
}

std::vector<arb::time_type> poisson_schedule_shim::events(arb::time_type t0, arb::time_type t1) {
    pyarb::assert_throw(is_nonneg()(t0), "t0 must be a non-negative number");
    pyarb::assert_throw(is_nonneg()(t1), "t1 must be a non-negative number");

    arb::schedule sched = poisson_schedule_shim::schedule();

    return as_vector(sched.events(t0, t1));
}

void register_schedules(py::module& m) {
    using namespace py::literals;
    using time_type = arb::time_type;

    py::class_<schedule_shim_base> schedule_base(m, "schedule_base", "Schedule abstract base class.");

    // Regular schedule
    py::class_<regular_schedule_shim, schedule_shim_base> regular_schedule(m, "regular_schedule",
        "Describes a regular schedule with multiples of dt within the interval [tstart, tstop).");

    regular_schedule
        .def(py::init<time_type, time_type, py::object>(),
            "tstart"_a, "dt"_a, "tstop"_a = py::none(),
            "Construct a regular schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence [ms].\n"
            "  dt:     The interval between time points [ms].\n"
            "  tstop:  No events delivered after this time [ms], None by default.")
        .def(py::init<time_type>(),
            "dt"_a,
            "Construct a regular schedule, starting from t = 0 and never terminating, with arguments:\n"
            "  dt:     The interval between time points [ms].\n")
        .def_property("tstart", &regular_schedule_shim::get_tstart, &regular_schedule_shim::set_tstart,
            "The delivery time of the first event in the sequence [ms].")
        .def_property("tstop", &regular_schedule_shim::get_tstop, &regular_schedule_shim::set_tstop,
            "No events delivered after this time [ms].")
        .def_property("dt", &regular_schedule_shim::get_dt, &regular_schedule_shim::set_dt,
            "The interval between time points [ms].")
        .def("events", &regular_schedule_shim::events,
            "A view of monotonically increasing time values in the half-open interval [t0, t1).")
        .def("__str__",  util::to_string<regular_schedule_shim>)
        .def("__repr__", util::to_string<regular_schedule_shim>);

    // Explicit schedule
    py::class_<explicit_schedule_shim, schedule_shim_base> explicit_schedule(m, "explicit_schedule",
        "Describes an explicit schedule at a predetermined (sorted) sequence of times.");

    explicit_schedule
        .def(py::init<>(),
            "Construct an empty explicit schedule.\n")
        .def(py::init<std::vector<time_type>>(),
            "times"_a,
            "Construct an explicit schedule with argument:\n"
            "  times: A list of times [ms], [] by default.")
        .def_property("times", &explicit_schedule_shim::get_times, &explicit_schedule_shim::set_times,
            "A list of times [ms].")
        .def("events", &explicit_schedule_shim::events,
            "A view of monotonically increasing time values in the half-open interval [t0, t1).")
        .def("__str__",  util::to_string<explicit_schedule_shim>)
        .def("__repr__", util::to_string<explicit_schedule_shim>);

    // Poisson schedule
    py::class_<poisson_schedule_shim, schedule_shim_base> poisson_schedule(m, "poisson_schedule",
        "Describes a schedule according to a Poisson process within the interval [tstart, tstop).");

    poisson_schedule
        .def(py::init<time_type, time_type, std::mt19937_64::result_type, py::object>(),
             "tstart"_a = 0., "freq"_a, "seed"_a = 0, "tstop"_a = py::none(),
            "Construct a Poisson schedule with arguments:\n"
            "  tstart: The delivery time of the first event in the sequence [ms], 0 by default.\n"
            "  freq:   The expected frequency [kHz].\n"
            "  seed:   The seed for the random number generator, 0 by default.\n"
            "  tstop:  No events delivered after this time [ms], None by default.")
        .def(py::init<time_type>(),
            "freq"_a,
            "Construct a Poisson schedule, starting from t = 0, default seed, with:\n"
            "  freq:   The expected frequency [kHz], 10 by default.\n")
        .def_property("tstart", &poisson_schedule_shim::get_tstart, &poisson_schedule_shim::set_tstart,
            "The delivery time of the first event in the sequence [ms].")
        .def_property("freq", &poisson_schedule_shim::get_freq, &poisson_schedule_shim::set_freq,
            "The expected frequency [kHz].")
        .def_readwrite("seed", &poisson_schedule_shim::seed,
            "The seed for the random number generator.")
        .def_property("tstop", &poisson_schedule_shim::get_tstop, &poisson_schedule_shim::set_tstop,
            "No events delivered after this time [ms].")
        .def("events", &poisson_schedule_shim::events,
            "A view of monotonically increasing time values in the half-open interval [t0, t1).")
        .def("__str__",  util::to_string<poisson_schedule_shim>)
        .def("__repr__", util::to_string<poisson_schedule_shim>);
}

}
