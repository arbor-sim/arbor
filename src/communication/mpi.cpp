#include <mpi.h>

#include <communication/mpi.hpp>

namespace arb {
namespace mpi {

// global guard for initializing mpi.

global_guard::global_guard(int *argc, char ***argv) {
    init(argc, argv);
}

global_guard::~global_guard() {
    finalize();
}

// MPI exception class.

mpi_error::mpi_error(const char* msg, int code):
    error_code_(code)
{
    thread_local char buffer[MPI_MAX_ERROR_STRING];
    int n;
    MPI_Error_string(error_code_, buffer, &n);
    message_ = "MPI error (";
    message_ += buffer;
    message_ += "): ";
    message_ += msg;
}

void handle_mpi_error(const char* msg, int code) {
    if (code!=MPI_SUCCESS) {
        throw mpi_error(msg, code);
    }
}

const char* mpi_error::what() const throw() {
    return message_.c_str();
}

int mpi_error::error_code() const {
    return error_code_;
}

void init(int *argc, char ***argv) {
    int provided;

    // initialize with thread serialized level of thread safety
    MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);

    if(provided<MPI_THREAD_SERIALIZED) {
        throw mpi_error("Unable to initialize MPI with MPI_THREAD_SERIALIZED", MPI_ERR_OTHER);
    }
}

void finalize() {
    MPI_Finalize();
}

int rank(MPI_Comm comm) {
    int r;
    handle_mpi_error("MPI_Rank", MPI_Comm_rank(comm, &r));
    return r;
}

int size(MPI_Comm comm) {
    int s;
    handle_mpi_error("MPI_Size", MPI_Comm_size(comm, &s));
    return s;
}

void barrier(MPI_Comm comm) {
    handle_mpi_error("MPI_Barrier", MPI_Barrier(comm));
}

} // namespace mpi
} // namespace arb
