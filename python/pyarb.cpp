#include <pybind11/pybind11.h>

// forward declarations of functions used to register API
// types and functions to be expose to python
namespace arb {
namespace py {

void register_identifiers(pybind11::module& m);
void register_contexts(pybind11::module& m);
void register_event_generators(pybind11::module& m);
void register_profilers(pybind11::module& m);

}
}

PYBIND11_MODULE(pyarb, m) {
    m.doc() = "pyarb: Python bindings for Arbor.";
    m.attr("__version__") = "dev";

    arb::py::register_identifiers(m);
    arb::py::register_contexts(m);
    arb::py::register_event_generators(m);
    arb::py::register_profilers(m);
}

