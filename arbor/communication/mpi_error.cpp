#include <mpi.h>

#include <arbor/communication/mpi_error.hpp>

namespace arb {

ARB_ARBOR_API const mpi_error_category_impl& mpi_error_category() {
    static mpi_error_category_impl the_category;
    return the_category;
}

const char* mpi_error_category_impl::name() const noexcept { return "MPI"; }

std::string mpi_error_category_impl::message(int ev) const {
    char err[MPI_MAX_ERROR_STRING];
    int r;
    MPI_Error_string(ev, err, &r);
    return err;
}

std::error_condition mpi_error_category_impl::default_error_condition(int ev) const noexcept {
    int eclass;
    MPI_Error_class(ev, &eclass);
    return std::error_condition(eclass, mpi_error_category());
}

} // namespace arb
