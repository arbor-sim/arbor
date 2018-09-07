#include <pybind11/pybind11.h>
#include <sstream>

#include <arbor/profile/meter_manager.hpp>
#include "context.hpp"

namespace pyarb {

void register_profilers(pybind11::module& m) {
    using namespace pybind11::literals;
    //
    // metering
    //
    pybind11::class_<arb::profile::meter_manager> meter_manager(m, "meter_manager");

    meter_manager
        .def(pybind11::init<>())

        // Need to use shimming of context due to pybind11 complaining about incomplete type of execution_context  
        .def("start", 
            [](arb::profile::meter_manager& manager, const context_shim& ctx){
                manager.start(ctx.context);
            },
            "context"_a,
            "Start profiling.")
        .def("checkpoint", 
            [](arb::profile::meter_manager& manager, std::string name, const context_shim& ctx){
                manager.checkpoint(name, ctx.context);
            },
            "name"_a,
            "context"_a,
            "Set checkpoint name.")
        .def("__str__", [](){return "<pyarb.meter_manager>";})
        .def("__repr__", [](){return "<pyarb.meter_manager>";});

    //
    // meter reporting
    //
    pybind11::class_<arb::profile::meter_report> meter_report(m, "meter_report");

    meter_report
        .def("__str__", [](const arb::profile::meter_report& report){
            std::stringstream s;
            s << report;
            return s.str();
        })
        .def("__repr__", [](const arb::profile::meter_report& report){
            std::stringstream s;
            s << report;
            return s.str();
        });

    m.def("make_meter_report", 
        [](const arb::profile::meter_manager& manager, const context_shim& ctx){
            return arb::profile::make_meter_report(manager, ctx.context);
        }, 
        "manager"_a,
        "context"_a,
        "Generate a meter_report from a set of meters.");
}

} // namespace pyarb
