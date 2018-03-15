#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cell.hpp>
#include <domain_decomposition.hpp>
#include <model.hpp>
#include <load_balance.hpp>
#include <profiling/meter_manager.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>

#include "cells.hpp"
#include "event_generator.hpp"
#include "lif_cell_description.hpp"
#include "print.hpp"
#include "recipe.hpp"
#include "sampling.hpp"

arb::domain_decomposition partition_load_balance(std::shared_ptr<arb::py::recipe>& r, const arb::hw::node_info& ni) {
    return arb::partition_load_balance(arb::py_recipe_shim(r), ni);
};

arb::domain_decomposition partition_load_balance(std::shared_ptr<arb::py::recipe>& r) {
    return arb::partition_load_balance(arb::py_recipe_shim(r), arb::hw::get_node_info());
};

// helpful string literals that reduce verbosity
using namespace pybind11::literals;

PYBIND11_MODULE(pyarb, m) {
    //
    // common types
    //

    pybind11::class_<arb::cell_member_type> cell_member(m, "cell_member",
        "For global identification of an item of cell local data.\n\n"
        "Items of cell_member_type must:\n"
        "(1) be associated with a unique cell, identified by the member gid;\n"
        "(2) identify an item within a cell-local collection by the member index.\n");

    cell_member
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](arb::cell_gid_type gid, arb::cell_lid_type idx) {
                arb::cell_member_type m;
                m.gid = gid;
                m.index = idx;
                return m;
            }))
        .def_readwrite("index", &arb::cell_member_type::index,
            "Cell-local index of the item.")
        .def_readwrite("gid",   &arb::cell_member_type::gid,
            "The global identifier of a cell.")
        .def("__str__",  &cell_member_string)
        .def("__repr__", &cell_member_string);

    //
    // spike recording
    //

    pybind11::class_<arb::spike> spike(m, "spike");
    spike
        .def(pybind11::init<>())
        .def_readwrite("source", &arb::spike::source)
        .def_readwrite("time", &arb::spike::time)
        .def("__str__",  &spike_string)
        .def("__repr__", &spike_string);

    // Use shared_ptr for spike_recorder, so that all copies of a recorder will
    // see the spikes from the model with which the recorder's callback has been
    // registered.
    pybind11::class_<spike_recorder, std::shared_ptr<spike_recorder>> spike_recorder(m, "spike_recorder");
    spike_recorder
        .def(pybind11::init<>())
        .def_property_readonly("spikes", [](const ::spike_recorder& s) {return *(s.spike_store.get());} );
    m.def("make_spike_recorder", &make_spike_recorder);

    //
    // cell types
    //

    // Wrap the cell_kind enum type.
    pybind11::enum_<arb::cell_kind>(m, "cell_kind")
        .value("cable1d", arb::cell_kind::cable1d_neuron)
        .value("regular_spike", arb::cell_kind::regular_spike_source)
        .value("data_spike", arb::cell_kind::data_spike_source)
        .value("lif", arb::cell_kind::lif_neuron);


    // Wrap the lif cell type.
    pybind11::class_<arb::lif_cell_description> lif_cell(m, "lif_cell");
    lif_cell
        .def(pybind11::init<>())
        .def_readwrite("tau_m", &arb::lif_cell_description::tau_m,
                "Membrane potential decaying constant [ms].")
        .def_readwrite("V_th", &arb::lif_cell_description::V_th,
                "Firing threshold [mV].")
        .def_readwrite("C_m", &arb::lif_cell_description::C_m,
                "Membrane capacitance [pF].")
        .def_readwrite("E_L", &arb::lif_cell_description::E_L,
                "Resting potential [mV].")
        .def_readwrite("V_m", &arb::lif_cell_description::V_m,
                "Initial value of the Membrane potential [mV].")
        .def_readwrite("V_reset", &arb::lif_cell_description::V_reset,
                "Reset potential [mV].")
        .def_readwrite("t_ref", &arb::lif_cell_description::t_ref,
                "Refractory period [ms].");

    // Wrap the regular spike source cell type.
    pybind11::class_<arb::rss_cell> rss_cell(m, "rss_cell");
    rss_cell.def(pybind11::init<>())
            .def_readwrite("start_time", &arb::rss_cell::start_time)
            .def_readwrite("period",     &arb::rss_cell::period)
            .def_readwrite("stop_time",  &arb::rss_cell::stop_time)
            .def("__str__",  &rss_cell_string)
            .def("__repr__", &rss_cell_string);

    // Wrap cell description type.
    pybind11::class_<arb::cell> cell(m, "cell");

    pybind11::class_<arb::segment_location> segment_location(m, "segment_location");
    segment_location
        .def(pybind11::init<arb::cell_lid_type, double>())
        .def_readwrite("segment", &arb::segment_location::segment)
        .def_readwrite("position", &arb::segment_location::position)
        .def("__str__",  &segment_location_string)
        .def("__repr__", &segment_location_string);

    // Don't expose underlying interface directly: instead use lamdbas to add some utility
    cell.def("add_synapse",
                [](arb::cell& c, arb::segment_location l)
                {c.add_synapse(l, arb::mechanism_spec("expsyn"));},
                "location"_a)
        .def("add_stimulus",
                [](arb::cell& c, arb::segment_location loc, double t0, double duration, double weight)
                {c.add_stimulus(loc, {t0, duration, weight});},
                "Add stimulus to the cell",
                "location"_a, "t0 (ms)"_a, "duration (ms)"_a, "weight (nA)"_a)
        .def("add_detector",  &arb::cell::add_detector,
                "location"_a, "threashold(mV)"_a)
        .def("__str__",  &cell_string)
        .def("__repr__", &cell_string);

    m.def("make_soma_cell", &make_cell_soma_only,
        "Make a single compartment cell with properties:"
        "\n    diameter 18.8 µm;"
        "\n    mechanisms HH;"
        "\n    bulk resistivitiy 100 Ω·cm;"
        "\n    capacitance 0.01 F⋅m⁻²." );

    //
    // Connections
    //

    pybind11::class_<arb::cell_connection> connection(m, "connection");

    connection
        .def(pybind11::init<>())
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, float, float>(),
            "source"_a, "destination"_a, "weight"_a, "delay"_a)
        .def_readwrite("source", &arb::cell_connection::source,
            "The source of the conection (type: pyarb.cell_member)")
        .def_readwrite("destination", &arb::cell_connection::dest,
            "The destination id of the connection (type: pyarb.cell_member)")
        .def_readwrite("weight", &arb::cell_connection::weight,
            "The weight of the connection (S⋅cm⁻²)")
        .def_readwrite("delay", &arb::cell_connection::delay,
            "The delay time of the connection (ms)")
        .def("__str__",  &connection_string)
        .def("__repr__", &connection_string);

    //
    // Event generators
    //

    pybind11::class_<arb::postsynaptic_spike_event>
        event(m, "event"
            "A postsynaptic spike event, delivered to a target at a specified time."
            " The target is typically a synapse. A postsynaptic event carries a"
            " floating point weight value whose units or interpretation is target dependent,"
            " though it is typically used to describe a change in conductance (S⋅cm⁻²).");

    event
        .def(pybind11::init<>())
        .def_readwrite("target", &arb::postsynaptic_spike_event::target,
            "Delivery target of the event (gid, lid)")
        .def_readwrite("time",   &arb::postsynaptic_spike_event::time,
            "Delivery time the event (ms)")
        .def_readwrite("weight", &arb::postsynaptic_spike_event::weight,
            "The weight of the event (S⋅cm⁻²)");

    // Bind the generic event_generator class
    pybind11::class_<arb::event_generator>
        event_generator(m, "event_generator");

    event_generator
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](const arb::py::sequence_generator_desc& d) {return d.make_cpp();}))
        .def(pybind11::init(
            [](const arb::py::poisson_generator_desc& d) {return d.make_cpp();}))
        .def(pybind11::init(
            [](const arb::py::regular_generator_desc& d) {return d.make_cpp();}))
        .def("next", &arb::event_generator::next,
            "Get the current event in the stream. \n"
            "Does not modify the state of the stream, i.e. multiple calls to \n"
            "next() will return the same event in the absence of calls to pop(), \n"
            "advance() or reset().")
        .def("pop", &arb::event_generator::pop,
            "Move the generator to the next event in the stream.")
        .def("reset", &arb::event_generator::reset,
            "Reset the generator to the same state that it had on construction.")
        .def("advance", &arb::event_generator::advance,
            "Update state of the generator such that the event returned by next() is \n"
            "the first event with delivery time≥t.");

    pybind11::class_<arb::py::regular_generator_desc> regular_generator(m, "regular_generator");
    regular_generator
        .def(pybind11::init<>())
        .def_readwrite("target", &arb::py::regular_generator_desc::target)
        .def_readwrite("weight", &arb::py::regular_generator_desc::weight)
        .def_readwrite("tstart", &arb::py::regular_generator_desc::tstart)
        .def_readwrite("tstop",  &arb::py::regular_generator_desc::tstop)
        .def_readwrite("dt",     &arb::py::regular_generator_desc::dt);

    pybind11::class_<arb::py::poisson_generator_desc> poisson_generator(m, "poisson_generator");
    poisson_generator
        .def(pybind11::init<>())
        .def_readwrite("target",      &arb::py::poisson_generator_desc::target)
        .def_readwrite("weight",      &arb::py::poisson_generator_desc::weight)
        .def_readwrite("tstart",      &arb::py::poisson_generator_desc::tstart)
        .def_readwrite("tstop",       &arb::py::poisson_generator_desc::tstop)
        .def_readwrite("rate_per_ms", &arb::py::poisson_generator_desc::rate_per_ms)
        .def_readwrite("seed",        &arb::py::poisson_generator_desc::seed);

    pybind11::class_<arb::py::sequence_generator_desc> sequence_generator(m, "sequence_generator");
    sequence_generator
        .def(pybind11::init<>())
        .def_readwrite("events", &arb::py::sequence_generator_desc::py_events);

    //
    // Recipes
    //

    pybind11::class_<arb::py::recipe,
                     arb::py::recipe_trampoline,
                     std::shared_ptr<arb::py::recipe>>
        recipe(m, "recipe", pybind11::dynamic_attr());

    recipe
        .def(pybind11::init<>())
        .def("num_cells", &arb::py::recipe::num_cells,
           "The number of cells in the model.")
        .def("cell_description",
           &arb::py::recipe::cell_description, pybind11::return_value_policy::copy,
           "High level decription of the cell with global identifier gid.")
        .def("kind", &arb::py::recipe::kind,
           "The cell_kind of cell with global identifier gid.")
        .def("connections_on", &arb::py::recipe::connections_on, "")
        .def("num_targets", &arb::py::recipe::num_targets, "")
        .def("num_sources", &arb::py::recipe::num_sources, "");

    //
    // Load balancing and domain decomposition
    //

    // tell python about the backend_kind enum type
    pybind11::enum_<arb::backend_kind>(m, "backend_kind")
        .value("gpu", arb::backend_kind::gpu)
        .value("multicore", arb::backend_kind::multicore);

    // group_description wrapper
    pybind11::class_<arb::group_description> group_description(m, "group_description");
    group_description
        .def(pybind11::init<arb::cell_kind, std::vector<arb::cell_gid_type>, arb::backend_kind>(),
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
    pybind11::class_<arb::domain_decomposition> domain_decomposition(m, "domain_decomposition");
    domain_decomposition
        .def(pybind11::init<>())
        .def("is_local_gid", &arb::domain_decomposition::is_local_gid,
            "Test if cell with gloabl identifier gid is in a local cell_group")
        .def_readwrite("num_domains", &arb::domain_decomposition::num_domains,
            "Number of distrubuted domains")
        .def_readwrite("domain_id", &arb::domain_decomposition::domain_id,
            "The index of the local domain")
        .def_readwrite("num_local_cells", &arb::domain_decomposition::num_local_cells,
            "Total number of cells in the local domain")
        .def_readwrite("num_global_cells", &arb::domain_decomposition::num_global_cells,
            "Total number of cells in the global model (sum over all domains)")
        .def_readwrite("groups", &arb::domain_decomposition::groups,
            "Descriptions of the cell groups on the local domain")
        .def("gid_domain",
            [](const arb::domain_decomposition& d, arb::cell_gid_type gid) {
                return d.gid_domain(gid);
            }, "The domain of cell with global identifier gid.", "gid"_a)
        .def("__str__",  &domain_decomposition_string)
        .def("__repr__", &domain_decomposition_string);

    // partition_load_balancer
    // The python recipe has to be shimmed for passing to the function that
    // takes a C++ recipe.
    m.def("partition_load_balance",
        static_cast<arb::domain_decomposition (*)(std::shared_ptr<arb::py::recipe>& r, const arb::hw::node_info& ni)>(
            &partition_load_balance
        ),
        "Simple load balancer.", "recipe"_a, "node"_a);
    m.def("partition_load_balance",
        static_cast<arb::domain_decomposition (*)(std::shared_ptr<arb::py::recipe>& r)>(
            &partition_load_balance
        ),
        "Simple load balancer.", "recipe"_a);

    // node_info which describes the resources on a compute node
    pybind11::class_<arb::hw::node_info> node_info(m, "node_info",
        "Describes the resources on a compute node.");
    node_info
        .def(pybind11::init<>())
        .def(pybind11::init<unsigned, unsigned>())
        .def_readwrite("num_cpu_cores", &arb::hw::node_info::num_cpu_cores,
                "The number of available CPU cores.")
        .def_readwrite("num_gpus", &arb::hw::node_info::num_gpus,
                "The number of available GPUs.")
        .def("__str__",  &node_info_string)
        .def("__repr__", &node_info_string);

    // get_node_info
    m.def("get_node_info", &arb::hw::get_node_info,
        "Returns a description of the hardware resources available on the host compute node.");

    //
    // models
    //
    pybind11::class_<arb::model> model(m, "model", "An Arbor model.");

    model
        // A custom constructor that wraps a python recipe with
        // arb::py_recipe_shim before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
                [](std::shared_ptr<arb::py::recipe>& r, const arb::domain_decomposition& d) {
                    return new arb::model(arb::py_recipe_shim(r), d);
                }),
                // Release the python gil, so that callbacks into the python
                // recipe r don't deadlock.
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Initialize the model described by a recipe, with cells and network "
                "distributed according to decomp.",
                "recipe"_a, "decomp"_a)
        .def("reset", &arb::model::reset,
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Reset the model to its initial state to rerun the simulation again.")
        .def("run", &arb::model::run,
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Advance the model state to time tfinal, in time steps of size dt.",
                "tfinal"_a, "dt"_a);

    //
    // metering
    //
    pybind11::class_<arb::util::meter_manager> meter_manager(m, "meter_manager");
    meter_manager.def(pybind11::init<>())
                     .def("start", &arb::util::meter_manager::start)
                     .def("checkpoint", &arb::util::meter_manager::checkpoint);

    // wrap meter_report type such that print(meter_report) works
    pybind11::class_<arb::util::meter_report> meter_report(m, "meter_report");
    meter_report.def("__str__", &meter_report_string)
                .def("__repr__",&meter_report_string);

    m.def("make_meter_report", &arb::util::make_meter_report,
          "Generate a meter_report from a set of meters.");

    m.def("profiler_output", &arb::util::profiler_output);
}
