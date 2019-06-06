#pragma once

#ifdef ARB_MPI_ENABLED
#include <mpi.h>

namespace pyarb {
// A shim is required for MPI_Comm, because OpenMPI defines it as a pointer to
// a forward-declared type, which pybind11 won't allow as an argument.
// MPICH and its derivatives use an integer.

struct mpi_comm_shim {
    MPI_Comm comm = MPI_COMM_WORLD;

    mpi_comm_shim() = default;
    mpi_comm_shim(MPI_Comm c): comm(c) {}

    mpi_comm_shim(pybind11::object o);
};

bool can_convert_to_mpi_comm(pybind11::object o);
MPI_Comm convert_to_mpi_comm(pybind11::object o);

} // namespace pyarb
#endif

