#pragma once

#include <arbor/version.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#endif

namespace sup {

int find_gpu();

#ifdef ARB_MPI_ENABLED
int find_gpu(MPI_Comm comm);
#endif

} // namespace sup

