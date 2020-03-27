#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/cable_cell.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>

#include "cells.hpp"
#include "conversion.hpp"
#include "error.hpp"
#include "event_generator.hpp"
#include "strprintf.hpp"
#include "recipe.hpp"

namespace pyarb {

// The py::recipe::cell_decription returns a pybind11::object, that is
// unwrapped and copied into a arb::util::unique_any.
arb::util::unique_any py_recipe_shim::get_cell_description(arb::cell_gid_type gid) const {
    return try_catch_pyexception(
                [&](){ return convert_cell(impl_->cell_description(gid)); },
                "Python error already thrown");
}

arb::cell_probe_address::probe_kind probe_kind_from_string(const std::string& name) {
    if (name == "voltage") {
        return arb::cell_probe_address::probe_kind::membrane_voltage;
    }
    else if (name == "current") {
        return arb::cell_probe_address::probe_kind::membrane_current;
    }
    else throw pyarb_error(util::pprintf("invalid probe kind: {}, neither voltage nor current", name));
}

arb::probe_info cable_probe(std::string kind, arb::cell_member_type id, arb::mlocation loc) {
    auto pkind = probe_kind_from_string(kind);
    arb::cell_probe_address probe{loc, pkind};
    return arb::probe_info{id, pkind, probe};
};

std::vector<arb::event_generator> convert_gen(std::vector<pybind11::object> pygens, arb::cell_gid_type gid) {
    using namespace std::string_literals;
    using pybind11::isinstance;
    using pybind11::cast;

    // Aquire the GIL because it must be held when calling isinstance and cast.
    pybind11::gil_scoped_acquire guard;

    std::vector<arb::event_generator> gens;
    gens.reserve(pygens.size());

    for (auto& g: pygens) {
        // check that a valid Python event_generator was passed.
        if (!isinstance<pyarb::event_generator_shim>(g)) {
            throw pyarb_error(
                util::pprintf(
                    "recipe supplied an invalid event generator for gid {}: {}", gid, pybind11::str(g)));
        }
        // get a reference to the python event_generator
        auto& p = cast<const pyarb::event_generator_shim&>(g);

        // convert the event_generator to an arb::event_generator
        gens.push_back(arb::schedule_generator({gid, p.target.index}, p.weight, std::move(p.time_sched)));
    }

    return gens;
}

std::vector<arb::event_generator> py_recipe_shim::event_generators(arb::cell_gid_type gid) const {
    return try_catch_pyexception(
                [&](){ return convert_gen(impl_->event_generators(gid), gid); },
                "Python error already thrown");
}

std::string con_to_string(const arb::cell_connection& c) {
    return util::pprintf("<arbor.connection: source ({},{}), destination ({},{}), delay {}, weight {}>",
         c.source.gid, c.source.index, c.dest.gid, c.dest.index, c.delay, c.weight);
}

std::string gj_to_string(const arb::gap_junction_connection& gc) {
    return util::pprintf("<arbor.gap_junction_connection: local ({},{}), peer ({},{}), ggap {}>",
         gc.local.gid, gc.local.index, gc.peer.gid, gc.peer.index, gc.ggap);
}

void register_recipe(pybind11::module& m) {
    using namespace pybind11::literals;

    // Connections
    pybind11::class_<arb::cell_connection> cell_connection(m, "connection",
        "Describes a connection between two cells:\n"
        "  Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively), a connection weight and a delay time.");
    cell_connection
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, float, float>(),
            "source"_a, "dest"_a, "weight"_a, "delay"_a,
            "Construct a connection with arguments:\n"
            "  source:      The source end point of the connection.\n"
            "  dest:        The destination end point of the connection.\n"
            "  weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).\n"
            "  delay:       The delay of the connection [ms].")
        .def_readwrite("source", &arb::cell_connection::source,
            "The source of the connection.")
        .def_readwrite("dest", &arb::cell_connection::dest,
            "The destination of the connection.")
        .def_readwrite("weight", &arb::cell_connection::weight,
            "The weight of the connection.")
        .def_readwrite("delay", &arb::cell_connection::delay,
            "The delay time of the connection [ms].")
        .def("__str__",  &con_to_string)
        .def("__repr__", &con_to_string);

    // Gap Junction Connections
    pybind11::class_<arb::gap_junction_connection> gap_junction_connection(m, "gap_junction_connection",
        "Describes a gap junction between two gap junction sites.");
    gap_junction_connection
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, double>(),
            "local"_a, "peer"_a, "ggap"_a,
            "Construct a gap junction connection with arguments:\n"
            "  local: One half of the gap junction connection.\n"
            "  peer:  Other half of the gap junction connection.\n"
            "  ggap:  Gap junction conductance [μS].")
        .def_readwrite("local", &arb::gap_junction_connection::local,
            "One half of the gap junction connection.")
        .def_readwrite("peer", &arb::gap_junction_connection::peer,
            "Other half of the gap junction connection.")
        .def_readwrite("ggap", &arb::gap_junction_connection::ggap,
            "Gap junction conductance [μS].")
        .def("__str__",  &gj_to_string)
        .def("__repr__", &gj_to_string);

    // Recipes
    pybind11::class_<py_recipe,
                     py_recipe_trampoline,
                     std::shared_ptr<py_recipe>>
        recipe(m, "recipe", pybind11::dynamic_attr(),
        "A description of a model, describing the cells and the network via a cell-centric interface.");
    recipe
        .def(pybind11::init<>())
        .def("num_cells", &py_recipe::num_cells, "The number of cells in the model.")
        .def("cell_description", &py_recipe::cell_description, pybind11::return_value_policy::copy,
            "gid"_a,
            "High level description of the cell with global identifier gid.")
        .def("cell_kind", &py_recipe::cell_kind,
            "gid"_a,
            "The kind of cell with global identifier gid.")
        .def("num_sources", &py_recipe::num_sources,
            "gid"_a,
            "The number of spike sources on gid, 0 by default.")
        .def("num_targets", &py_recipe::num_targets,
            "gid"_a,
            "The number of post-synaptic sites on gid, 0 by default.")
        .def("num_gap_junction_sites", &py_recipe::num_gap_junction_sites,
            "gid"_a,
            "The number of gap junction sites on gid, 0 by default.")
        .def("event_generators", &py_recipe::event_generators,
            "gid"_a,
            "A list of all the event generators that are attached to gid, [] by default.")
        .def("connections_on", &py_recipe::connections_on,
            "gid"_a,
            "A list of all the incoming connections to gid, [] by default.")
        .def("gap_junctions_on", &py_recipe::gap_junctions_on,
            "gid"_a,
            "A list of the gap junctions connected to gid, [] by default.")
        .def("num_probes", &py_recipe::num_probes,
            "gid"_a,
            "The number of probes on gid, 0 by default.")
        .def("get_probe", &py_recipe::get_probe,
            "id"_a,
            "The probe(s) to allow monitoring, must be provided if num_probes() returns a non-zero value.")
        // TODO: py_recipe::global_properties
        .def("__str__",  [](const py_recipe&){return "<arbor.recipe>";})
        .def("__repr__", [](const py_recipe&){return "<arbor.recipe>";});

    // Probes
    m.def("cable_probe", &cable_probe,
        "Description of a probe at a location on a cable cell with id available for monitoring data of kind "\
        "where kind is one of 'voltage' or 'current'.",
        "kind"_a, "id"_a, "location"_a);

    pybind11::class_<arb::probe_info> probe(m, "probe");
    probe
        .def("__repr__", [](const arb::probe_info& p){return util::pprintf("<arbor.probe: cell {}, probe {}>", p.id.gid, p.id.index);})
        .def("__str__",  [](const arb::probe_info& p){return util::pprintf("<arbor.probe: cell {}, probe {}>", p.id.gid, p.id.index);});
}
} // namespace pyarb
