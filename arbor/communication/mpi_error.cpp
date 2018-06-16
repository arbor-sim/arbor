#include <mpi.h>

#include <arbor/communication/mpi_error.hpp>

namespace arb {

const mpi_error_category_impl& mpi_error_category() {
    static mpi_error_category_impl the_category;
    return the_category;
}

}
