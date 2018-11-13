#include <pybind11/pybind11.h>
#include <sstream>

#include <arbor/profile/meter_manager.hpp>
#include "context.hpp"

namespace pyarb {

void register_profilers(pybind11::module& m) {
    using namespace pybind11::literals;
    pybind11::class_<arb::profile::meter_manager> meter_manager(m, "meter_manager");

    meter_manager
        .def(pybind11::init<>())

        .def("start",
            [](arb::profile::meter_manager& manager, const context_shim& ctx){
                manager.start(ctx.context);
            },
            "context"_a,
            "Start the metering. Records a time stamp, that marks the start of the first checkpoint timing region.")
        .def("checkpoint",
            [](arb::profile::meter_manager& manager, std::string name, const context_shim& ctx){
                manager.checkpoint(name, ctx.context);
            },
            "name"_a,
            "context"_a,
            "Create a new checkpoint. Records the time since the last checkpoint (or the call to start if no previous checkpoints), and restarts the timer for the next checkpoint..")
        .def("__str__", [](){return "<pyarb.meter_manager>";})
        .def("__repr__", [](){return "<pyarb.meter_manager>";});

    pybind11::class_<arb::profile::meter_report> meter_report(m, "meter_report");

    auto to_string = [](const arb::profile::meter_report& report) {
        std::stringstream s;
        s << report;
        return s.str();
    };

    meter_report
        .def("__str__", to_string)
        .def("__repr__", to_string);

    m.def("make_meter_report",
        [](const arb::profile::meter_manager& manager, const context_shim& ctx){
            return arb::profile::make_meter_report(manager, ctx.context);
        },
        "manager"_a,
        "context"_a,
        "Generate a meter report.");
}

} // namespace pyarb
