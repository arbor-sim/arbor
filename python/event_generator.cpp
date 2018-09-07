#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

#include "event_generator.hpp"

namespace pyarb {

template <typename Sched>
event_generator py_make_event_generator(
        arb::cell_lid_type lid,
        double weight,
        const Sched& sched)
{
    return event_generator(lid, weight, sched.schedule());
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

    pybind11::class_<event_generator> event_generator(m, "event_generator");

    event_generator
        .def(pybind11::init<>(
            [](arb::cell_lid_type lid, double weight, const regular_schedule_shim& sched){
                return py_make_event_generator(lid, weight, sched);}))
        .def(pybind11::init<>(
            [](arb::cell_lid_type lid, double weight, const explicit_schedule_shim& sched){
                return py_make_event_generator(lid, weight, sched);}))
        ;

}

} // namespace pyarb
