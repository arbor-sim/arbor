#include <mpi.h>

#include "communication/mpi.hpp"

namespace arb {
namespace mpi {

int rank(MPI_Comm comm) {
    int r;
    MPI_OR_THROW(MPI_Comm_rank, comm, &r);
    return r;
}

int size(MPI_Comm comm) {
    int s;
    MPI_OR_THROW(MPI_Comm_size, comm, &s);
    return s;
}

void barrier(MPI_Comm comm) {
    MPI_OR_THROW(MPI_Barrier, comm);
}

} // namespace mpi
} // namespace arb
