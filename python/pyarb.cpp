#include <pybind11/pybind11.h>

// forward declarations of functions used to register API
// types and functions to be expose to python
namespace arb {
namespace py {

void register_cells(pybind11::module& m);
void register_contexts(pybind11::module& m);
void register_domain_decomposition(pybind11::module& m);
void register_event_generators(pybind11::module& m);
void register_identifiers(pybind11::module& m);
void register_recipe(pybind11::module& m);
void register_simulation(pybind11::module& m);
void register_spike_handling(pybind11::module& m);

}
}

PYBIND11_MODULE(pyarb, m) {
    m.doc() = "pyarb: Python bindings for Arbor.";
    m.attr("__version__") = "dev";

    arb::py::register_contexts(m);
    arb::py::register_cells(m);
    arb::py::register_domain_decomposition(m);
    arb::py::register_event_generators(m);
    arb::py::register_identifiers(m);
    arb::py::register_recipe(m);
    arb::py::register_simulation(m);
    arb::py::register_spike_handling(m);
}

