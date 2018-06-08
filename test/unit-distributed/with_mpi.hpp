#pragma once

#include <mpi.h>

#include <communication/mpi_error.hpp>

struct with_mpi {
    with_mpi(int& argc, char**& argv, bool fatal_errors = true) {
        init(&argc, &argv, fatal_errors);
    }

    with_mpi(bool fatal_errors = true) {
        init(nullptr, nullptr, fatal_errors);
    }

    ~with_mpi() {
        MPI_Finalize();
    }

private:
    void init(int* argcp, char*** argvp, bool fatal_errors) {
        int provided;
        int ev = MPI_Init_thread(argcp, argvp, MPI_THREAD_SERIALIZED, &provided);
        if (ev) {
            throw arb::mpi_error(ev, "MPI_Init_thread");
        }
        else if (provided<MPI_THREAD_SERIALIZED) {
            throw arb::mpi_error(MPI_ERR_OTHER, "MPI_Init_thread: MPI_THREAD_SERIALIZED unsupported");
        }

        if (!fatal_errors) {
            MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        }
    }
};
