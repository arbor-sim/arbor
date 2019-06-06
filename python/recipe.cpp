#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>

#include "error.hpp"
#include "event_generator.hpp"
#include "strprintf.hpp"
#include "recipe.hpp"

namespace pyarb {

// The py::recipe::cell_decription returns a pybind11::object, that is
// unwrapped and copied into a arb::util::unique_any.

arb::util::unique_any py_recipe_shim::get_cell_description(arb::cell_gid_type gid) const {
    using pybind11::isinstance;
    using pybind11::cast;

    // Aquire the GIL because it must be held when calling isinstance and cast.
    auto guard = pybind11::gil_scoped_acquire();

    // Get the python object pyarb::cell_description from the python front end
    pybind11::object o = impl_->cell_description(gid);

    if (isinstance<arb::cable_cell>(o)) {
        return arb::util::unique_any(cast<arb::cable_cell>(o));
    }

    else if (isinstance<arb::lif_cell>(o)) {
        return arb::util::unique_any(cast<arb::lif_cell>(o));
    }

    else if (isinstance<arb::spike_source_cell>(o)) {
        return arb::util::unique_any(cast<arb::spike_source_cell>(o));
    }

    else if (isinstance<arb::benchmark_cell>(o)) {
        return arb::util::unique_any(cast<arb::benchmark_cell>(o));
    }

    throw pyarb_error(
                        "recipe.cell_description returned \""
                        + std::string(pybind11::str(o))
                        + "\" which does not describe a known Arbor cell type");
}

// The py::recipe::global_properties returns a pybind11::object, that is
// unwrapped and copied into a arb::util::any.
arb::util::any py_recipe_shim::get_global_properties(arb::cell_kind kind) const {
    using pybind11::cast;

    // Aquire the GIL because it must be held when calling cast.
    auto guard = pybind11::gil_scoped_acquire();

    // Get the python object pyarb::global_properties from the python front end
    pybind11::object o = impl_->global_properties(kind);

    if (kind == arb::cell_kind::cable) {
        return arb::util::any(cast<arb::cable_cell_global_properties>(o));
    }

    else return arb::util::any{};

    throw pyarb_error( "recipe.global_properties returned \""
                       + std::string(pybind11::str(o))
                       + "\" which does not describe a known Arbor global property description");

}

std::vector<arb::event_generator> py_recipe_shim::event_generators(arb::cell_gid_type gid) const {
    using namespace std::string_literals;
    using pybind11::isinstance;
    using pybind11::cast;

    // Aquire the GIL because it must be held when calling isinstance and cast.
    auto guard = pybind11::gil_scoped_acquire();

    // Get the python list of pyarb::event_generator_shim from the python front end.
    auto pygens = impl_->event_generators(gid);

    std::vector<arb::event_generator> gens;
    gens.reserve(pygens.size());

    for (auto& g: pygens) {
        // check that a valid Python event_generator was passed.
        if (!isinstance<pyarb::event_generator_shim>(g)) {
            std::stringstream s;
            s << "recipe supplied an invalid event generator for gid "
            << gid << ": " << pybind11::str(g);
            throw pyarb_error(s.str());
        }
        // get a reference to the python event_generator
        auto& p = cast<const pyarb::event_generator_shim&>(g);

        // convert the event_generator to an arb::event_generator
        gens.push_back(arb::schedule_generator({gid, p.target.index}, p.weight, std::move(p.time_sched)));
    }

    return gens;
}

// TODO: implement py_recipe_shim::probe_info

std::string con_to_string(const arb::cell_connection& c) {
    return util::pprintf("<connection: ({},{}) -> ({},{}), delay {}, weight {}>",
         c.source.gid, c.source.index, c.dest.gid, c.dest.index, c.delay, c.weight);
}

std::string gj_to_string(const arb::gap_junction_connection& gc) {
    return util::pprintf("<gap junction: ({},{}) <-> ({},{}), conductance {}>",
         gc.local.gid, gc.local.index, gc.peer.gid, gc.peer.index, gc.ggap);
}

void register_recipe(pybind11::module& m) {
    using namespace pybind11::literals;

    // Connections
    pybind11::class_<arb::cell_connection> cell_connection(m, "cell_connection",
        "Describes a connection between two cells:\n"
        "a pre-synaptic source and a post-synaptic destination.");
    cell_connection
        .def(pybind11::init<>(
            [](){return arb::cell_connection({0u,0u}, {0u,0u}, 0.f, 0.f);}),
            "Construct a connection with default arguments:\n"
            "  source:      gid 0, index 0.\n"
            "  destination: gid 0, index 0.\n"
            "  weight:      0.\n"
            "  delay:       0 ms.\n")
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, float, float>(),
            "source"_a, "destination"_a, "weight"_a = 0.f, "delay"_a = 0.f,
            "Construct a connection with arguments:\n"
            "  source:      The source end point of the connection.\n"
            "  destination: The destination end point of the connection.\n"
            "  weight:      The weight delivered to the target synapse (dimensionless with interpretation specific to synapse type of target, default 0.).\n"
            "  delay:       The delay of the connection (unit: ms, default 0.).\n")
        .def_readwrite("source", &arb::cell_connection::source,
            "The source of the connection.")
        .def_readwrite("destination", &arb::cell_connection::dest,
            "The destination of the connection.")
        .def_readwrite("weight", &arb::cell_connection::weight,
            "The weight of the connection (unit: S⋅cm⁻²).")
        .def_readwrite("delay", &arb::cell_connection::delay,
            "The delay time of the connection (unit: ms).")
        .def("__str__",  &con_to_string)
        .def("__repr__", &con_to_string);

    // Gap Junction Connections
    pybind11::class_<arb::gap_junction_connection> gap_junction_connection(m, "gap_junction_connection",
        "Describes a gap junction between two gap junction sites.");
    gap_junction_connection
        .def(pybind11::init<>(
            [](){return arb::gap_junction_connection({0u,0u}, {0u,0u}, 0.f);}),
            "Construct a gap junction connection with default arguments:\n"
            "  local: gid 0, index 0.\n"
            "  peer:  gid 0, index 0.\n"
            "  ggap:  0 μS.\n")
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, double>(),
            "local"_a, "peer"_a, "ggap"_a = 0.f,
            "Construct a gap junction connection with arguments:\n"
            "  local: One half of the gap junction connection.\n"
            "  peer:  Other half of the gap junction connection.\n"
            "  ggap:  Gap junction conductance (unit: μS, default 0.).\n")
        .def_readwrite("local", &arb::gap_junction_connection::local,
            "One half of the gap junction connection.")
        .def_readwrite("peer", &arb::gap_junction_connection::peer,
            "Other half of the gap junction connection.")
        .def_readwrite("ggap", &arb::gap_junction_connection::ggap,
            "Gap junction conductance (unit: μS).")
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
            "The number of spike sources on gid (default 0).")
        .def("num_targets", &py_recipe::num_targets,
            "gid"_a,
            "The number of post-synaptic sites on gid (default 0).")
        // TODO: py_recipe::num_probes
        .def("num_gap_junction_sites", &py_recipe::num_gap_junction_sites,
            "gid"_a,
            "The number of gap junction sites on gid (default 0).")
        .def("event_generators", &py_recipe::event_generators,
            "gid"_a,
            "A list of all the event generators that are attached to gid (default []).")
        .def("connections_on", &py_recipe::connections_on,
            "gid"_a,
            "A list of all the incoming connections to gid (default []).")
        .def("gap_junctions_on", &py_recipe::gap_junctions_on,
            "gid"_a,
            "A list of the gap junctions connected to gid (default []).")
        // TODO: py_recipe::get_probe
        .def("global_properties", &py_recipe::global_properties, pybind11::return_value_policy::copy,
            "cell_kind"_a,
            "Global property type specific to a given cell kind.")
        .def("__str__", [](const py_recipe&){return "<pyarb.recipe>";})
        .def("__repr__", [](const py_recipe&){return "<pyarb.recipe>";});
}
} // namespace pyarb
