#include <stdexcept>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>
#include <arbor/util/optional.hpp>

#include "conversion.hpp"
#include "error.hpp"
#include "event_generator.hpp"
#include "schedule.hpp"

namespace pyarb {

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
