#include <sstream>
#include <string>

#include <pybind11/pybind11.h>

#include <arbor/version.hpp>

#include "strprintf.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>

#include <arbor/communication/mpi_error.hpp>

#include "mpi.hpp"

#ifdef ARB_WITH_MPI4PY
#include <mpi4py/mpi4py.h>
#endif
#endif

namespace pyarb {

#ifdef ARB_MPI_ENABLED

// Convert a Python object an MPI Communicator.
// Used to construct mpi_comm_shim from arbitrary Python types.
// Currently only supports mpi4py communicators, but could be extended to
// other types.
MPI_Comm convert_to_mpi_comm(pybind11::object o) {
#ifdef ARB_WITH_MPI4PY
    import_mpi4py();
    if (PyObject_TypeCheck(o.ptr(), &PyMPIComm_Type)) {
        return *PyMPIComm_Get(o.ptr());
    }
#endif
    throw arb::mpi_error(MPI_ERR_OTHER, "Invalid MPI Communicatior");
}

mpi_comm_shim::mpi_comm_shim(pybind11::object o) {
    comm = convert_to_mpi_comm(o);
}

// Test if a Python object can be converted to an mpi_comm_shim.
bool can_convert_to_mpi_comm(pybind11::object o) {
#ifdef ARB_WITH_MPI4PY
    import_mpi4py();
    if (PyObject_TypeCheck(o.ptr(), &PyMPIComm_Type)) {
        return true;
    }
#endif
    return false;
}

// Some helper functions for initializing and finalizing MPI.
// Arbor requires at least MPI_THREAD_SERIALIZED, because the communication task
// that uses MPI can be run on any thread, and there will never be more than one
// concurrent communication task.

void mpi_init() {
    int provided = MPI_THREAD_SINGLE;
    int ev = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
    if (ev) {
        throw arb::mpi_error(ev, "MPI_Init_thread");
    }
    else if (provided<MPI_THREAD_SERIALIZED) {
        throw arb::mpi_error(MPI_ERR_OTHER, "MPI_Init_thread: MPI_THREAD_SERIALIZED unsupported");
    }
}

void mpi_finalize() {
    MPI_Finalize();
}

int mpi_is_initialized() {
    int initialized;
    MPI_Initialized(&initialized);
    return initialized;
}

int mpi_is_finalized() {
    int finalized;
    MPI_Finalized(&finalized);
    return finalized;
}
// Define the stringifier for mpi_comm_shim here, to minimise the ifdefication
// elsewhere in this wrapper code.

std::ostream& operator<<(std::ostream& o, const mpi_comm_shim& c) {
    if (c.comm==MPI_COMM_WORLD) {
        return o << "<arbor.mpi_comm: MPI_COMM_WORLD>";
    }
    else {
        return o << "<arbor.mpi_comm: " << c.comm << ">";
    }
}

void register_mpi(pybind11::module& m) {
    using namespace std::string_literals;
    using namespace pybind11::literals;

    pybind11::class_<mpi_comm_shim> mpi_comm(m, "mpi_comm");
    mpi_comm
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](pybind11::object o){ return mpi_comm_shim(o); }),
            "mpi_comm_obj"_a, "MPI communicator object.")
        .def("__str__",  util::to_string<mpi_comm_shim>)
        .def("__repr__", util::to_string<mpi_comm_shim>);

    m.def("mpi_init", &mpi_init, "Initialize MPI with MPI_THREAD_SINGLE, as required by Arbor.");
    m.def("mpi_finalize", &mpi_finalize, "Finalize MPI (calls MPI_Finalize)");
    m.def("mpi_is_initialized", &mpi_is_initialized, "Check if MPI is initialized.");
    m.def("mpi_is_finalized", &mpi_is_finalized, "Check if MPI is finalized.");
}
#endif
} // namespace pyarb
