#include <sstream>
#include <string>

#include <arbor/version.hpp>

#include <pybind11/pybind11.h>

#ifdef ARB_MPI_ENABLED
#include <arbor/communication/mpi_error.hpp>

#include <mpi.h>

#include "mpi.hpp"

#ifdef ARB_WITH_MPI4PY
#include <mpi4py/mpi4py.h>
#endif

namespace pyarb {

#ifdef ARB_WITH_MPI4PY

mpi_comm_shim comm_from_mpi4py(pybind11::object& o) {
    import_mpi4py();

    // If object is already a mpi4py communicator, return
    if (PyObject_TypeCheck(o.ptr(), &PyMPIComm_Type)) {
        return mpi_comm_shim(*PyMPIComm_Get(o.ptr()));
    }
    throw arb::mpi_error(MPI_ERR_OTHER, "The argument is not an mpi4py communicator");
}

#endif

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

std::string mpi_comm_string(const mpi_comm_shim& c) {
    std::stringstream s;

    s << "<mpi_comm: ";
    if (c.comm==MPI_COMM_WORLD) s << "MPI_COMM_WORLD>";
    else s << c.comm << ">";
    return s.str();
}

void register_mpi(pybind11::module& m) {
    using namespace std::string_literals;

    pybind11::class_<mpi_comm_shim> mpi_comm(m, "mpi_comm");
    mpi_comm
        .def(pybind11::init<>())
        .def("__str__", &mpi_comm_string)
        .def("__repr__", &mpi_comm_string);

    m.def("mpi_init", &mpi_init, "Initialize MPI with MPI_THREAD_SINGLE, as required by Arbor.");
    m.def("mpi_finalize", &mpi_finalize, "Finalize MPI (calls MPI_Finalize)");
    m.def("mpi_is_initialized", &mpi_is_initialized, "Check if MPI is initialized.");
    m.def("mpi_is_finalized", &mpi_is_finalized, "Check if MPI is finalized.");

    #ifdef ARB_WITH_MPI4PY
    m.def("mpi_comm_from_mpi4py", comm_from_mpi4py);
    #endif
}

} // namespace pyarb
#endif

