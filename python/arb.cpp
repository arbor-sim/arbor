#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <domain_decomposition.hpp>
#include <model.hpp>
#include <load_balance.hpp>
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

// helpful string literals that reduce verbosity
using namespace pybind11::literals;

PYBIND11_MODULE(arb, m) {
    //
    // util types
    //

    // util::any
    pb::class_<arb::util::any> any(m, "any");
    any.def("__str__",  &any_string)
       .def("__repr__", &any_string);

    // register the bad_any_cast exception as TypeError
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
          .def("num_cells", &arb::recipe::num_cells,
               "The number of cells in the model.")
          .def("get_cell_description", &arb::recipe::get_cell_description,
               "High level decription of the cell with global identifier gid.")
          .def("get_cell_kind", &arb::recipe::get_cell_kind,
               "The cell_kind of cell with global identifier gid.");

    //
    // load balancing and domain decomposition
    //

    // tell python about the backend_kind enum type
    pybind11::enum_<arb::backend_kind>(m, "backend_kind")
        .value("gpu", arb::backend_kind::gpu)
        .value("multicore", arb::backend_kind::multicore);

    // group_description wrapper
    pb::class_<arb::group_description> group_description(m, "group_description");
    group_description
        .def(pb::init<arb::cell_kind, std::vector<arb::cell_gid_type>, arb::backend_kind>(),
            "construct group_description with cell_kind, list of gids, and backend.")
        .def_readonly("kind", &arb::group_description::kind,
            "The type of cell in the cell group.")
        .def_readonly("gids", &arb::group_description::gids,
            "The gids of the cells in the group in ascending order.")
        .def_readonly("backend", &arb::group_description::backend,
            "The hardware backend on which the cell group will run.")
        .def("__str__",  &group_description_string)
        .def("__repr__", &group_description_string);

    // domain_decomposition wrapper
    pb::class_<arb::domain_decomposition> domain_decomposition(m, "domain_decomposition");
    domain_decomposition
        .def(pb::init<>())
        .def("is_local_gid", &arb::domain_decomposition::is_local_gid,
            "Test if cell with gloabl identifier gid is in a local cell_group")
        .def_readonly("num_domains", &arb::domain_decomposition::num_domains,
            "Number of distrubuted domains")
        .def_readonly("domain_id", &arb::domain_decomposition::domain_id,
            "The index of the local domain")
        .def_readonly("num_local_cells", &arb::domain_decomposition::num_local_cells,
            "Total number of cells in the local domain")
        .def_readonly("num_global_cells", &arb::domain_decomposition::num_global_cells,
            "Total number of cells in the global model (sum over all domains)")
        .def_readonly("groups", &arb::domain_decomposition::groups,
            "Descriptions of the cell groups on the local domain")
        .def("__str__",  &domain_decomposition_string)
        .def("__repr__", &domain_decomposition_string);

        /// Return the domain id of cell with gid.
        /// Supplied by the load balancing algorithm that generates the domain
        /// decomposition.
        //std::function<int(cell_gid_type)> gid_domain;

    // TODO: write guard types for MPI state.

    // TODO: wrap this in a helper function that automatically makes the node description
    m.def("partition_load_balance", &arb::partition_load_balance,
        "Simple load balancer.");

    // node_info which describes the resources on a compute node
    pb::class_<arb::hw::node_info> node_info(m, "node_info", "Describes the resources on a compute node.");
    node_info
        .def(pb::init<>())
        .def(pb::init<unsigned, unsigned>())
        .def_readwrite("num_cpu_cores", &arb::hw::node_info::num_cpu_cores, "The number of available CPU cores.")
        .def_readwrite("num_gpus", &arb::hw::node_info::num_gpus, "The number of available GPUs.")
        .def("__str__",  &node_info_string)
        .def("__repr__", &node_info_string);

    // get_node_info
    m.def("get_node_info", &arb::hw::get_node_info,
        "Returns a description of the hardware resources available on the host compute node.");

    //
    // models
    //
    pb::class_<arb::model> model(m, "model", "An Arbor model.");

    model
        .def(pb::init<const arb::recipe&, const arb::domain_decomposition&>())
        .def("reset", &arb::model::reset,
            "Reset the model to its initial state to rerun the simulation again.")
        .def("run", &arb::model::run,
            "Advance the model state to a future time.", "tfinal"_a, "dt"_a);

        //model(const recipe& rec, const domain_decomposition& decomp);
        //void reset();
        //time_type run(time_type tfinal, time_type dt);

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

