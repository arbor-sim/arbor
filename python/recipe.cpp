#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>

#include "event_generator.hpp"
#include "recipe.hpp"
#include "strings.hpp"

namespace arb {
namespace py {

// The py::recipe::cell_decription returns a pybind11::object, that is
// unwrapped and copied into a util::unique_any.
util::unique_any py_recipe_shim::get_cell_description(cell_gid_type gid) const {
    auto guard = pybind11::gil_scoped_acquire();
    pybind11::object o = impl_->cell_description(gid);

    if (pybind11::isinstance<arb::mc_cell>(o)) {
        return util::unique_any(pybind11::cast<arb::mc_cell>(o));
    }
//  if (pybind11::isinstance<lif_cell>(o)) {
//      return util::unique_any(pybind11::cast<lif_cell>(o));
//  }

    throw arb::python_error(
        "recipe.cell_description returned \""
        + std::string(pybind11::str(o))
        + "\" which does not describe a known Arbor cell type.");
}

std::vector<arb::event_generator> py_recipe_shim::event_generators(cell_gid_type gid) const {
    using namespace std::string_literals;
    using pybind11::isinstance;
    using pybind11::cast;

    // Aquire the GIL because it must be held when calling isinstance and cast.
    auto guard = pybind11::gil_scoped_acquire();

    // Get the python list of arb::py::event_generator from the python front end.
    auto pygens = impl_->event_generators(gid);

    std::vector<arb::event_generator> gens;
    gens.reserve(pygens.size());

    for (auto& g: pygens) {
        // check that a valid Python event_generator was passed.
        if (!isinstance<arb::py::event_generator>(g)) {
            std::stringstream s;
            s << "recipe supplied an invalid event generator for gid "
              << gid << ": " << pybind11::str(g);
            throw python_error(s.str());
        }
        // get a reference to the python event_generator
        auto& p = cast<const arb::py::event_generator&>(g);

        // convert the event_generator to an arb::event_generator
        gens.push_back(
            arb::schedule_generator(
                {gid, p.lid}, p.weight, std::move(p.time_seq)));
    }

    return gens;
}

template <typename Sched>
arb::benchmark_cell py_make_benchmark_cell(const Sched& sched)
{
    return arb::benchmark_cell(sched.schedule(), 1.0);
}

void register_recipe(pybind11::module& m) {
    using namespace pybind11::literals;

    // Wrap the cell_kind enum type.
    pybind11::enum_<arb::cell_kind>(m, "cell_kind")
        .value("benchmark", arb::cell_kind::benchmark)
        .value("cable1d", arb::cell_kind::cable1d_neuron)
        .value("lif", arb::cell_kind::lif_neuron)
        .value("spike_souce", arb::cell_kind::spike_source);

    // Connections
    pybind11::class_<arb::cell_connection> connection(m, "connection");

    connection
        .def(pybind11::init<>(
            [](){return arb::cell_connection({0u,0u}, {0u,0u}, 0.f, 0.f);}))
        .def(pybind11::init<arb::cell_member_type, arb::cell_member_type, float, float>(),
            "source"_a, "destination"_a, "weight"_a, "delay"_a)
        .def_readwrite("source", &arb::cell_connection::source,
            "The source of the conection (type: pyarb.cell_member)")
        .def_readwrite("destination", &arb::cell_connection::dest,
            "The destination of the connection (type: pyarb.cell_member)")
        .def_readwrite("weight", &arb::cell_connection::weight,
            "The weight of the connection (S⋅cm⁻²)")
        .def_readwrite("delay", &arb::cell_connection::delay,
            "The delay time of the connection (ms)")
        .def("__str__", &connection_string)
        .def("__repr__", &connection_string);

    // Recipies
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
            "gid"_a,
            "The cell_kind of cell with global identifier gid.")
        .def("connections_on", &arb::py::recipe::connections_on,
            "gid"_a,
            "A list of the incoming connections to gid")
        .def("num_targets", &arb::py::recipe::num_targets,
            "gid"_a,
            "The number of event targets on gid (e.g. synapses)")
        .def("num_sources", &arb::py::recipe::num_sources,
            "gid"_a,
            "The number of spike sources on gid")
        .def("__str__", [](const arb::py::recipe&){return "<pyarb.recipe>";})
        .def("__repr__", [](const arb::py::recipe&){return "<pyarb.recipe>";});

    // Cell kinds.

    pybind11::class_<arb::benchmark_cell> benchmark_cell(m, "benchmark_cell");

    benchmark_cell
        .def(pybind11::init<>())
        .def_readwrite("realtime_ratio", &arb::benchmark_cell::realtime_ratio,
            "Time taken in ms to advance the cell one ms of simulation time. \n"
            "If equal to 1, then a single cell can be advanced in realtime ");
    /*
    struct benchmark_cell {
        // Describes the time points at which spikes are to be generated.
        schedule time_sequence;

        // Time taken in ms to advance the cell one ms of simulation time.
        // If equal to 1, then a single cell can be advanced in realtime 
        double realtime_ratio;
    };
    */

}

} // namespace py
} // namespace arb

