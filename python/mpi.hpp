#pragma once

#include <sstream>
#include <string>

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/version.hpp>

#include <pybind11/pybind11.h>

#ifdef ARB_MPI_ENABLED
#include <arbor/communication/mpi_error.hpp>
#include <mpi.h>
#endif

#ifdef ARB_MPI_ENABLED

namespace pyarb {
// A shim is required for MPI_Comm, because OpenMPI defines it as a pointer to
// a forward-declared type, which pybind11 won't allow as an argument.
// MPICH and its derivatives use an integer.

struct mpi_comm_shim {
    MPI_Comm comm = MPI_COMM_WORLD;

    mpi_comm_shim() = default;
    mpi_comm_shim(MPI_Comm c): comm(c) {}
};

} // namespace pyarb
#endif

