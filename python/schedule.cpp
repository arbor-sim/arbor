#include <arbor/schedule.hpp>
#include <arbor/common_types.hpp>
#include <arbor/util/optional.hpp>

#include <pybind11/pybind11.h>

#include "conversion.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"

namespace pyarb {

std::ostream& operator<<(std::ostream& o, const regular_schedule_shim& x) {
    return o << "<regular_schedule: tstart "
             << x.tstart << " ms, dt "
             << x.dt << " ms, tstop "
             << x.tstop << " ms>";
}

std::ostream& operator<<(std::ostream& o, const explicit_schedule_shim& e) {
    o << "<explicit_schedule: times [";
    return util::csv(o, e.times) << "] ms>";
};

std::ostream& operator<<(std::ostream& o, const poisson_schedule_shim& p) {
    return o << "<poisson_schedule: tstart " << p.tstart << " ms"
             << ", freq " << p.freq << " Hz"
             << ", seed " << p.seed << ">";
};

//
// regular_schedule shim
//

regular_schedule_shim::regular_schedule_shim(
        pybind11::object t0,
        time_type deltat,
        pybind11::object t1)
{
    set_tstart(t0);
    set_tstop(t1);
    set_dt(deltat);
}

void regular_schedule_shim::set_tstart(pybind11::object t) {
    tstart = py2optional<time_type>(
            t, "tstart must a non-negative number, or None", is_nonneg());
};

void regular_schedule_shim::set_tstop(pybind11::object t) {
    tstop = py2optional<time_type>(
            t, "tstop must a non-negative number, or None", is_nonneg());
};

void regular_schedule_shim::set_dt(arb::time_type delta_t) {
    pyarb::assert_throw(is_nonneg()(delta_t), "dt must be a non-negative number");
    dt = delta_t;
};

regular_schedule_shim::opt_time_type regular_schedule_shim::get_tstart() const {
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
            tstart.value_or(arb::terminal_time),
            dt,
            tstop.value_or(arb::terminal_time));
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
                "explicit time schedule can not contain negative values");
    }
};

std::vector<arb::time_type> explicit_schedule_shim::get_times() const {
    return times;
}

arb::schedule explicit_schedule_shim::schedule() const {
    return arb::explicit_schedule(times);
}

//
// poisson_schedule shim
//

poisson_schedule_shim::poisson_schedule_shim(
        arb::time_type ts,
        arb::time_type f,
        rng_type::result_type s)
{
    set_tstart(ts);
    set_freq(f);
    seed = s;
}

void poisson_schedule_shim::set_tstart(arb::time_type t) {
    pyarb::assert_throw(is_nonneg()(t), "tstart must be a non-negative number");
    tstart = t;
};

void poisson_schedule_shim::set_freq(arb::time_type f) {
    pyarb::assert_throw(is_nonneg()(f), "frequency must be a non-negative number");
    freq = f;
};

arb::time_type poisson_schedule_shim::get_tstart() const {
    return tstart;
}

arb::time_type poisson_schedule_shim::get_freq() const {
    return freq;
}

arb::schedule poisson_schedule_shim::schedule() const {
    // convert frequency to kHz.
    return arb::poisson_schedule(tstart, freq/1000., rng_type(seed));
}

void register_schedules(pybind11::module& m) {
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
        .def("__str__",  util::to_string<regular_schedule_shim>)
        .def("__repr__", util::to_string<regular_schedule_shim>);

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
        .def("__str__",  util::to_string<explicit_schedule_shim>)
        .def("__repr__", util::to_string<explicit_schedule_shim>);

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
        .def("__str__",  util::to_string<poisson_schedule_shim>)
        .def("__repr__", util::to_string<poisson_schedule_shim>);
}


}
