#pragma once

#include <exception>

#include <mpi.h>

#include <arbor/communication/mpi_error.hpp>

namespace arbenv {

struct with_mpi {
    with_mpi(int& argc, char**& argv, bool fatal_errors = true) {
        init(&argc, &argv, fatal_errors);
    }

    explicit with_mpi(bool fatal_errors = true) {
        init(nullptr, nullptr, fatal_errors);
    }

    ~with_mpi() {
        // Test if the stack is being unwound because of an exception.
        // If other ranks have not thrown an exception, there is a very
        // high likelihood that the MPI_Finalize will hang due to the other
        // ranks calling other MPI calls.
        // We don't directly call MPI_Abort in this case because that would
        // force exit the application before the exception that is unwinding
        // the stack has been caught, which would deny the opportunity to print
        // an error message explaining the cause of the exception.
        if (std::uncaught_exceptions()==0) {
            MPI_Finalize();
        }
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

} // namespace arbenv
