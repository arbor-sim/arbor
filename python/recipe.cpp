#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/recipe.hpp>

#include "arbor/cable_cell_param.hpp"
#include "error.hpp"
#include "event_generator.hpp"
#include "strprintf.hpp"
#include "recipe.hpp"

namespace pyarb {

namespace U = arb::units;

// Convert a cell description inside a Python object to a cell description in a
// unique_any, as required by the recipe interface.
// This helper is only to be called while holding the GIL. We require this guard
// across the lifetime of the description object `o`, since this function can be
// called without holding the GIL, ie from `simulation::init`, and `o` is a
// python object that can only be destroyed while holding the GIL. The fact that
// `cell_description` has a scoped GIL does not help with destruction as it
// happens outside that scope. `Description` needs to be extended in Python,
// inheriting from the pybind11 class.
static arb::util::unique_any convert_cell(pybind11::object o) {
    using pybind11::isinstance;
    using pybind11::cast;

    if (isinstance<arb::spike_source_cell>(o)) {
        return arb::util::unique_any(cast<arb::spike_source_cell>(o));
    }
    if (isinstance<arb::benchmark_cell>(o)) {
        return arb::util::unique_any(cast<arb::benchmark_cell>(o));
    }
    if (isinstance<arb::lif_cell>(o)) {
        return arb::util::unique_any(cast<arb::lif_cell>(o));
    }
    if (isinstance<arb::cable_cell>(o)) {
        return arb::util::unique_any(cast<arb::cable_cell>(o));
    }

    throw pyarb_error("recipe.cell_description returned \""
                      + std::string(pybind11::str(o))
                      + "\" which does not describe a known Arbor cell type");
}

// The py::recipe::cell_decription returns a pybind11::object, that is
// unwrapped and copied into a arb::util::unique_any.
 arb::util::unique_any recipe_shim::get_cell_description(arb::cell_gid_type gid) const {
    return try_catch_pyexception([&](){
        pybind11::gil_scoped_acquire guard;
        return convert_cell(impl_->cell_description(gid));
    },
        "Python error already thrown");
}

// Convert global properties inside a Python object to a
// std::any, as required by the recipe interface.
// This helper is only to called while holding the GIL, see above.
static std::any convert_gprop(pybind11::object o) {
    if (o.is(pybind11::none())) {
        return {};
    }
    if (!pybind11::isinstance<arb::cable_cell_global_properties>(o)) {
        throw pyarb_error("recipe.global_properties must return a valid description of cable cell properties of type arbor.cable_global_properties");
    }
    return pybind11::cast<arb::cable_cell_global_properties>(o);
}

// The py::recipe::global_properties returns a pybind11::object, that is
// unwrapped and copied into an std::any.
std::any recipe_shim::get_global_properties(arb::cell_kind kind) const {
    return try_catch_pyexception([&](){
        pybind11::gil_scoped_acquire guard;
        return convert_gprop(impl_->global_properties(kind));
    },
    "Python error already thrown");
}

// This helper is only to called while holding the GIL, see above.
static std::vector<arb::event_generator> convert_gen(std::vector<pybind11::object> pygens, arb::cell_gid_type gid) {
    using namespace std::string_literals;
    using pybind11::isinstance;
    using pybind11::cast;

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
        gens.push_back(arb::event_generator(p.target, p.weight, std::move(p.time_sched)));
    }

    return gens;
}

std::vector<arb::event_generator> recipe_shim::event_generators(arb::cell_gid_type gid) const {
    return try_catch_pyexception([&](){
        pybind11::gil_scoped_acquire guard;
        return convert_gen(impl_->event_generators(gid), gid);
    },
    "Python error already thrown");
}

std::string con_to_string(const arb::cell_connection& c) {
    return util::pprintf("<arbor.connection: source ({}, \"{}\", {}), destination (\"{}\", {}), delay {}, weight {}>",
         c.source.gid, c.source.label.tag, c.source.label.policy, c.target.tag, c.target.policy, c.delay, c.weight);
}

std::string raw_to_string(const arb::raw_cell_connection& c) {
    return util::pprintf("<arbor.raw_connection: source ({}, {}), destination (\"{}\", {}), delay {}, weight {}>",
         c.source.gid, c.source.index, c.target.tag, c.target.policy, c.delay, c.weight);
}


std::string gj_to_string(const arb::gap_junction_connection& gc) {
    return util::pprintf("<arbor.gap_junction_connection: peer ({}, \"{}\", {}), local (\"{}\", {}), weight {}>",
         gc.peer.gid, gc.peer.label.tag, gc.peer.label.policy, gc.local.tag, gc.local.policy, gc.weight);
}

void register_recipe(pybind11::module& m) {
    using namespace pybind11::literals;

    // Connections
    pybind11::class_<arb::cell_connection> cell_connection(m, "connection",
        "Describes a connection between two cells:\n"
        "  Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively), a connection weight and a delay time.");
    cell_connection
        .def(pybind11::init<arb::cell_global_label_type, arb::cell_local_label_type, float, const U::quantity&>(),
            "source"_a, "dest"_a, "weight"_a, "delay"_a,
            "Construct a connection with arguments:\n"
            "  source:      The source end point of the connection.\n"
            "  dest:        The destination end point of the connection.\n"
            "  weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).\n"
            "  delay:       The delay of the connection [ms].")
        .def_readwrite("source", &arb::cell_connection::source,
            "The source gid and label of the connection.")
        .def_readwrite("dest", &arb::cell_connection::target,
            "The destination label of the connection.")
        .def_readwrite("weight", &arb::cell_connection::weight,
            "The weight of the connection.")
        .def_readwrite("delay", &arb::cell_connection::delay,
            "The delay time of the connection [ms].")
        .def("__str__",  &con_to_string)
        .def("__repr__", &con_to_string);

    pybind11::class_<arb::raw_cell_connection> raw_cell_connection(m, "connection",
        "Describes a manually resolved connection between two cells:\n"
        "  Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively), a connection weight and a delay time.");
    raw_cell_connection
        .def(pybind11::init<arb::cell_member_type, arb::cell_local_label_type, float, const U::quantity&>(),
            "source"_a, "dest"_a, "weight"_a, "delay"_a,
            "Construct a connection with arguments:\n"
            "  source:      The source end point of the connection.\n"
            "  dest:        The destination end point of the connection.\n"
            "  weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).\n"
            "  delay:       The delay of the connection [ms].")
        .def_readwrite("source", &arb::raw_cell_connection::source,
            "The source gid and index of the connection.")
        .def_readwrite("dest", &arb::raw_cell_connection::target,
            "The destination label of the connection.")
        .def_readwrite("weight", &arb::raw_cell_connection::weight,
            "The weight of the connection.")
        .def_readwrite("delay", &arb::raw_cell_connection::delay,
            "The delay time of the connection [ms].")
        .def("__str__",  &con_to_string)
        .def("__repr__", &con_to_string);

    pybind11::class_<arb::ext_cell_connection> ext_cell_connection(m, "external_connection",
        "Describes a connection between two cells located in Arbor and an external simulation:\n"
        "  Defined by pre-synaptic and post-synaptic endpoints, a connection weight and a delay time.");
    ext_cell_connection
        .def(pybind11::init<arb::cell_remote_label_type, arb::cell_local_label_type, float, const U::quantity&>(),
            "source"_a, "dest"_a, "weight"_a, "delay"_a,
            "Construct a connection with arguments:\n"
            "  source:      The source end point of the connection.\n"
            "  dest:        The destination end point of the connection.\n"
            "  weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).\n"
            "  delay:       The delay of the connection [ms].")
        .def_readwrite("source", &arb::ext_cell_connection::source,
            "The source gid and label of the connection.")
        .def_readwrite("dest", &arb::ext_cell_connection::target,
            "The destination label of the connection.")
        .def_readwrite("weight", &arb::ext_cell_connection::weight,
            "The weight of the connection.")
        .def_readwrite("delay", &arb::ext_cell_connection::delay,
            "The delay time of the connection [ms].")
        .def("__str__",  &con_to_string)
        .def("__repr__", &con_to_string);


    // Gap Junction Connections
    pybind11::class_<arb::gap_junction_connection> gap_junction_connection(m, "gap_junction_connection",
        "Describes a gap junction between two gap junction sites.");
    gap_junction_connection
        .def(pybind11::init<arb::cell_global_label_type, arb::cell_local_label_type, double>(),
            "peer"_a, "local"_a, "weight"_a,
            "Construct a gap junction connection with arguments:\n"
            "  peer:  remote half of the gap junction connection.\n"
            "  local: local half of the gap junction connection.\n"
            "  weight:  Gap junction connection weight [unit-less].")
        .def_readwrite("peer", &arb::gap_junction_connection::peer,
            "Remote gid and label of the gap junction connection.")
        .def_readwrite("local", &arb::gap_junction_connection::local,
            "Local label of the gap junction connection.")
        .def_readwrite("weight", &arb::gap_junction_connection::weight,
            "Gap junction connection weight [unit-less].")
        .def("__str__",  &gj_to_string)
        .def("__repr__", &gj_to_string);

    // Recipes
    pybind11::class_<recipe,
                     recipe_trampoline,
                     std::shared_ptr<recipe>>
        recipe(m, "recipe", pybind11::dynamic_attr(),
        "A description of a model, describing the cells and the network via a cell-centric interface.");
    recipe
        .def(pybind11::init<>())
        .def("num_cells", &recipe::num_cells, "The number of cells in the model.")
        .def("cell_description", &recipe::cell_description, pybind11::return_value_policy::copy,
            "gid"_a,
            "High level description of the cell with global identifier gid.")
        .def("cell_kind",
             &recipe::cell_kind,
            "gid"_a,
            "The kind of cell with global identifier gid.")
        .def("event_generators", &recipe::event_generators,
            "gid"_a,
            "A list of all the event generators that are attached to gid, [] by default.")
        .def("connections_on", &recipe::connections_on,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
            "gid"_a,
            "A list of all the incoming connections to gid, [] by default.")
        .def("resolve_sources", &recipe::resolve_sources,
            "Global string label resolution enabled?.")
        .def("external_connections_on", &recipe::external_connections_on,
            "gid"_a,
            "A list of all the incoming connections from _remote_ locations to gid, [] by default.")
        .def("gap_junctions_on", &recipe::gap_junctions_on,
            "gid"_a,
            "A list of the gap junctions connected to gid, [] by default.")
        .def("network_description", &recipe::network_description,
            "Network description of cell connections.")
        .def("cell_isometry", &recipe::cell_isometry,
            "Isometry describing translation and rotation of cell.")
        .def("probes", &recipe::probes,
            "gid"_a,
            "The probes to allow monitoring.")
        .def("global_properties", &recipe::global_properties,
            "kind"_a,
            "The default properties applied to all cells of type 'kind' in the model.")
        .def("__str__",  [](const ::pyarb::recipe&){return "<arbor.recipe>";})
        .def("__repr__", [](const ::pyarb::recipe&){return "<arbor.recipe>";});
}
} // namespace pyarb
