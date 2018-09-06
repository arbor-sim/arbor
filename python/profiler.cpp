#include <pybind11/pybind11.h>

#include <arbor/profile/meter_manager.hpp>
#include "context.hpp"

namespace arb {
namespace py {

void register_profilers(pybind11::module& m) {
    //
    // metering
    //
    pybind11::class_<arb::profile::meter_manager> meter_manager(m, "meter_manager");

    meter_manager
        .def(pybind11::init<>())

        // Need to use shimming of context due to pybind11 complaining about incomplete type of execution_cntext  
        .def("start", [](arb::profile::meter_manager& manager, const arb::py::context_shim& ctx){\
            manager.start(ctx.context);\
        })
        .def("checkpoint", [](arb::profile::meter_manager& manager, std::string name, const arb::py::context_shim& ctx){\
            manager.checkpoint(name, ctx.context);
        })
        .def("__str__", [](){return "<pyarb.meter_manager>";})
        .def("__repr__", [](){return "<pyarb.meter_manager>";});

    //
    // meter reporting
    //
    pybind11::class_<arb::profile::meter_report> meter_report(m, "meter_report");

    meter_report
        .def("__str__", [](){return "<pyarb.meter_report>";})
        .def("__repr__", [](){return "<pyarb.meter_report>";});

    m.def("make_meter_report", [](const arb::profile::meter_manager& manager, const arb::py::context_shim& ctx){\
        return arb::profile::make_meter_report(manager, ctx.context);
        }, "Generate a meter_report from a set of meters.");
}

} // namespace py
} // namespace arb
