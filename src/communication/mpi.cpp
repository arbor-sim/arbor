#include <mpi.h>

#include <communication/mpi.hpp>

namespace nest {
namespace mc {
namespace mpi {

// global state
namespace state {
    int size = 1;
    int rank = 0;
} // namespace state

void init(int *argc, char ***argv) {
    int provided;

    // initialize with thread serialized level of thread safety
    PE("MPI", "Init");
    MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
    assert(provided>=MPI_THREAD_SERIALIZED);
    PL(2);

    PE("rank-size");
    MPI_Comm_rank(MPI_COMM_WORLD, &state::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state::size);
    PL();
}

void finalize() {
    PE("MPI", "Finalize");
    MPI_Finalize();
    PL(2);
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

    PE("MPI", "Allreduce-ballot");
    MPI_Allreduce(&value, &result, 1, traits::mpi_type(), MPI_LAND, MPI_COMM_WORLD);
    PL(2);

    return result;
}

} // namespace mpi
} // namespace mc
} // namespace nest
