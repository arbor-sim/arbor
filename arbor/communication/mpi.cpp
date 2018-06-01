#include <mpi.h>

#include <communication/mpi.hpp>

namespace arb {
namespace mpi {

// global state
namespace state {
    // TODO: in case MPI_Init is never called, this will mimic one rank with rank 0.
    // This is not ideal: calls to MPI-dependent features such as reductions will
    // still fail, however this allows us to run all the unit tests until the
    // run-time executors are implemented.
    int size = 1;
    int rank = 0;
} // namespace state

void init(int *argc, char ***argv) {
    int provided;

    // initialize with thread serialized level of thread safety
    MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    assert(provided>=MPI_THREAD_SERIALIZED);

    MPI_Comm_rank(MPI_COMM_WORLD, &state::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state::size);
}

void finalize() {
    MPI_Finalize();
}

bool is_root() {
    return state::rank == 0;
}

int rank() {
    return state::rank;
}

int size() {
    return state::size;
}

void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

bool ballot(bool vote) {
    using traits = mpi_traits<char>;

    char result;
    char value = vote ? 1 : 0;

    MPI_Allreduce(&value, &result, 1, traits::mpi_type(), MPI_LAND, MPI_COMM_WORLD);

    return result;
}

} // namespace mpi
} // namespace arb
