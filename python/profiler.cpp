#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/version.hpp>

#include "context.hpp"
#include "strprintf.hpp"

namespace pyarb {

void register_profiler(pybind11::module& m) {
    using namespace pybind11::literals;

    // meter manager
    pybind11::class_<arb::profile::meter_manager> meter_manager(m, "meter_manager",
        "Manage metering by setting checkpoints and starting the timing region.");
    meter_manager
        .def(pybind11::init<>())
        .def("start",
            [](arb::profile::meter_manager& manager, const context_shim& ctx){
                manager.start(ctx.context);
            },
            "context"_a,
            "Start the metering. Records a time stamp,\
             that marks the start of the first checkpoint timing region.")
        .def("checkpoint",
            [](arb::profile::meter_manager& manager, std::string name, const context_shim& ctx){
                manager.checkpoint(name, ctx.context);
            },
            "name"_a, "context"_a,
            "Create a new checkpoint. Records the time since the last checkpoint\
             (or the call to start if no previous checkpoints exist),\
             and restarts the timer for the next checkpoint.")
        .def_property_readonly("checkpoint_names", &arb::profile::meter_manager::checkpoint_names,
            "A list of all metering checkpoint names.")
        .def_property_readonly("times", &arb::profile::meter_manager::times,
            "A list of all metering times.")
        .def("__str__",  [](const arb::profile::meter_manager&){return "<arbor.meter_manager>";})
        .def("__repr__", [](const arb::profile::meter_manager&){return "<arbor.meter_manager>";});

    // meter report
    pybind11::class_<arb::profile::meter_report> meter_report(m, "meter_report",
        "Summarises the performance meter results, used to print a report to screen or file.\n"
        "If a distributed context is used, the report will contain a summary of results from all MPI ranks.");
    meter_report
        .def(pybind11::init<>(
        [](const arb::profile::meter_manager& manager, const context_shim& ctx){
            return arb::profile::make_meter_report(manager, ctx.context);
            }),
            "manager"_a, "context"_a)
        .def("__str__",  [](arb::profile::meter_report& r){return util::pprintf("{}", r);})
        .def("__repr__", [](arb::profile::meter_report& r){return "<arbor.meter_report>";});

#ifdef ARB_PROFILE_ENABLED
    m.def("profiler_initialize", [](context_shim& ctx) {
        arb::profile::profiler_initialize(ctx.context);
    });
    m.def("profiler_summary", [](){
        std::stringstream stream;
        stream << arb::profile::profiler_summary();
        return stream.str();
    });
#endif
}

} // namespace pyarb
