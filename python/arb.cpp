#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <profiling/meter_manager.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>

#include "print.hpp"
#include "recipe.hpp"

namespace pb = pybind11;

template <typename T>
arb::util::any wrap_any(T value) {
    return arb::util::any(std::move(value));
}

PYBIND11_MODULE(arb, m) {
    //
    // util types
    //

    pb::class_<arb::util::any> any(m, "any");
    any.def("__str__",  &any_string)
       .def("__repr__", &any_string);

    pb::register_exception<arb::util::bad_any_cast>(m, "TypeError");

    //
    // cell types
    //

    // tell python about the cell_kind enum type
    pybind11::enum_<arb::cell_kind>(m, "cell_kind")
        .value("cable1d", arb::cell_kind::cable1d_neuron)
        .value("regular_spike", arb::cell_kind::regular_spike_source)
        .value("data_spike", arb::cell_kind::data_spike_source);

    // wrap the regular spike source cell type
    pb::class_<arb::rss_cell> rss_cell(m, "rss_cell");
    rss_cell.def(pb::init<>())
            .def_readwrite("start_time", &arb::rss_cell::start_time)
            .def_readwrite("period",     &arb::rss_cell::period)
            .def_readwrite("stop_time",  &arb::rss_cell::stop_time)
            .def("__str__",  &rss_cell_string)
            .def("__repr__", &rss_cell_string)
            .def("wrap", &wrap_any<arb::rss_cell>);

    //
    // recipes
    //
    pb::class_<arb::recipe, arb::py_recipe> recipe(m, "recipe");
    recipe.def(pb::init<>())
          .def("num_cells", &arb::recipe::num_cells)
          .def("get_cell_description", &arb::recipe::get_cell_description)
          .def("get_cell_kind", &arb::recipe::get_cell_kind);

    //
    // load balancing and domain decomposition
    //

    //
    // models
    //

    //
    // metering
    //
    pb::class_<arb::util::measurement> measurement(m, "measurement",
             "Describes the recording of a single statistic over the course of a simulation,\ngathered by the meter_manager.");
    measurement.def_readwrite("name", &arb::util::measurement::name,
                    "Descriptive label of the measurement, e.g. 'wall time' or 'memory'.")
               .def_readwrite("units", &arb::util::measurement::units,
                    "SI units of the measurement, e.g. s or MiB.")
               .def_readwrite("measurements", &arb::util::measurement::measurements,
                    "A list of measurements, with one entry for each checkpoint.\n"
                    "Each entry is a list of values, with one value for each domain (MPI rank).");

    pb::class_<arb::util::meter_manager> meter_manager(m, "meter_manager");
    meter_manager.def(pb::init<>())
                     .def("start", &arb::util::meter_manager::start)
                     .def("checkpoint", &arb::util::meter_manager::checkpoint);

    // wrap meter_report type such that print(meter_report) works
    pb::class_<arb::util::meter_report> meter_report(m, "meter_report");
    meter_report.def("__str__", &meter_report_string)
                .def("__repr__",&meter_report_string);

    m.def("make_meter_report", &arb::util::make_meter_report,
          "Generate a meter_report from a set of meters.");
}

