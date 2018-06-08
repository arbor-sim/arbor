#pragma once

#include <system_error>

#include <mpi.h>

namespace arb {
enum mpi_error_code {};
enum mpi_error_condition {
};
}

namespace std {
template <> struct is_error_code_enum<enum mpi_error_code>: true_type {};
template <> struct is_error_condition_enum<enum mpi_error_condition>: true_type {};
}

namespace arb {

class mpi_error_category_impl: public std::error_category {
    const char* name() const override { return "MPI"; }
    std::string message(int ev) const override;
};

const mpi_error_category_impl& mpi_error_category();

struct mpi_error: std::system_error {
    explicit mpi_error(int mpi_err):
        std::system_error(mpi_err, mpi_error_category()) {}

    mpi_error(int mpi_err, const std::string& what_arg):
        std::system_error(mpi_err, mpi_error_category(), what_arg) {}
};

} // namespace arb


