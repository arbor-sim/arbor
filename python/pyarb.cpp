#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <sstream>

#include <arbor/spike.hpp>
#include <arbor/common_types.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/version.hpp>

#include "pyarb.hpp"

// Forward declarations of functions used to register API
// types and functions to be exposed to Python.
namespace pyarb {

void register_cable_loader(pybind11::module& m);
void register_cable_probes(pybind11::module& m, pyarb_global_ptr);
void register_cells(pybind11::module& m);
void register_config(pybind11::module& m);
void register_contexts(pybind11::module& m);
void register_domain_decomposition(pybind11::module& m);
void register_event_generators(pybind11::module& m);
void register_identifiers(pybind11::module& m);
void register_mechanisms(pybind11::module& m);
void register_morphology(pybind11::module& m);
void register_profiler(pybind11::module& m);
void register_recipe(pybind11::module& m);
void register_schedules(pybind11::module& m);
void register_simulation(pybind11::module& m, pyarb_global_ptr);
void register_single_cell(pybind11::module& m);
void register_arborenv(pybind11::module& m);

#ifdef ARB_MPI_ENABLED
void register_mpi(pybind11::module& m);
#endif

} // namespace pyarb

PYBIND11_MODULE(_arbor, m) {
    pyarb::pyarb_global_ptr global_ptr(new pyarb::pyarb_global);

    // Register NumPy structured datatypes for Arbor structures used in NumPy array outputs.
    PYBIND11_NUMPY_DTYPE(arb::cell_member_type, gid, index);
    PYBIND11_NUMPY_DTYPE(arb::spike, source, time);

    m.doc() = "arbor: multicompartment neural network models.";
    m.attr("__version__") = ARB_VERSION;

    pyarb::register_cable_loader(m);
    pyarb::register_cable_probes(m, global_ptr);
    pyarb::register_cells(m);
    pyarb::register_config(m);
    pyarb::register_contexts(m);
    pyarb::register_domain_decomposition(m);
    pyarb::register_event_generators(m);
    pyarb::register_identifiers(m);
    pyarb::register_mechanisms(m);
    pyarb::register_morphology(m);
    pyarb::register_profiler(m);
    pyarb::register_recipe(m);
    pyarb::register_schedules(m);
    pyarb::register_simulation(m, global_ptr);
    pyarb::register_single_cell(m);
    pyarb::register_arborenv(m);

    // This is the fallback. All specific translators take precedence by being
    // registered *later*.
    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        }
        catch (const arb::arbor_exception& e) {
            std::stringstream msg;
            msg << e.what()
                << "\n"
                << e.where;
            PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
        }
        catch (const arb::arbor_internal_error& e) {
            std::stringstream msg;
            msg << e.what()
                << "\n"
                << e.where;
            PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
        }
    });

    // Translate Arbor errors -> Python exceptions.
    pybind11::register_exception<arb::file_not_found_error>(m, "FileNotFoundError", PyExc_FileNotFoundError);
    pybind11::register_exception<arb::zero_thread_requested_error>(m, "ValueError", PyExc_ValueError);


    #ifdef ARB_MPI_ENABLED
    pyarb::register_mpi(m);
    #endif
}
