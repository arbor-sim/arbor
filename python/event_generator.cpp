#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

#include "event_generator.hpp"

namespace arb {
namespace py {

// A Python shim that holds the information that describes an
// arb::regular_schedule. This is wrapped in pybind11, and users constructing
// a regular_schedule in python are manipulating this type. This is converted to
// an arb::regular_schedule when a C++ recipe is created from a Python recipe.
struct regular_schedule_shim {
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
        return regular_schedule(tstart, dt, tstop);
    }
};

// A Python shim that holds the information that describes an
// arb::explicit_schedule. This is wrapped in pybind11, and users constructing
// a explicit_schedule in python are manipulating this type. This is converted to
// an arb::explicit_schedule when a C++ recipe is created from a Python recipe.
struct explicit_schedule_shim {
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

        return explicit_schedule(times);
    }
};


// make_event_generator is an overloaded function for converting 

template <typename Sched>
arb::py::event_generator py_make_event_generator(
        arb::cell_lid_type lid,
        double weight,
        const Sched& sched)
{
    return arb::py::event_generator(lid, weight, sched.schedule());
}

void register_event_generators(pybind11::module& m) {
    //
    // time sequence wrappers
    //

    pybind11::class_<regular_schedule_shim> regular_schedule(m, "regular_schedule");

    regular_schedule
        .def(pybind11::init<>())
        .def(pybind11::init<arb::time_type, arb::time_type, arb::time_type>())
        .def_readwrite("tstart", &regular_schedule_shim::tstart,
                "First time in sequence (ms).")
        .def_readwrite("tstop",  &regular_schedule_shim::tstop,
                "Latest possible time in sequence (ms).")
        .def_readwrite("dt",     &regular_schedule_shim::dt,
                "Distance between time points in sequence (ms).");

    pybind11::class_<explicit_schedule_shim> explicit_schedule(m, "explicit_schedule");

    explicit_schedule
        .def(pybind11::init<>())
        .def_readwrite("times", &explicit_schedule_shim::py_times,
                "A list of times in the schedule (ms).");

    //
    // event_generator
    //

    pybind11::class_<arb::py::event_generator> event_generator(m, "event_generator");

    event_generator
        .def(pybind11::init<>(
            [](arb::cell_lid_type lid, double weight, const regular_schedule_shim& sched){
                return py_make_event_generator(lid, weight, sched);}))
        .def(pybind11::init<>(
            [](arb::cell_lid_type lid, double weight, const explicit_schedule_shim& sched){
                return py_make_event_generator(lid, weight, sched);}))
        ;

}

} // namespace arb
} // namespace py
