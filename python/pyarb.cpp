#include <pybind11/pybind11.h>

#include <arbor/version.hpp>

// Forward declarations of functions used to register API
// types and functions to be exposed to Python.
namespace pyarb {

void register_cells(pybind11::module& m);
void register_config(pybind11::module& m);
void register_contexts(pybind11::module& m);
void register_domain_decomposition(pybind11::module& m);
void register_event_generators(pybind11::module& m);
void register_identifiers(pybind11::module& m);
void register_recipe(pybind11::module& m);
void register_schedules(pybind11::module& m);
void register_simulation(pybind11::module& m);
void register_spike_handling(pybind11::module& m);

#ifdef ARB_MPI_ENABLED
void register_mpi(pybind11::module& m);
#endif
}

PYBIND11_MODULE(arbor, m) {
    m.doc() = "arbor: Python bindings for Arbor.";
    m.attr("__version__") = ARB_VERSION;

    pyarb::register_cells(m);
    pyarb::register_config(m);
    pyarb::register_contexts(m);
    pyarb::register_domain_decomposition(m);
    pyarb::register_event_generators(m);
    pyarb::register_identifiers(m);
    #ifdef ARB_MPI_ENABLED
    pyarb::register_mpi(m);
    #endif
    pyarb::register_recipe(m);
    pyarb::register_schedules(m);
    pyarb::register_simulation(m);
    pyarb::register_spike_handling(m);
}
