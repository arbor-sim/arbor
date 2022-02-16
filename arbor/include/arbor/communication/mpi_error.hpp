#pragma once

#include <string>
#include <system_error>

#include <mpi.h>

#include <arbor/export.hpp>

namespace arb {

enum class mpi_errc {
    success = MPI_SUCCESS,
    invalid_buffer = MPI_ERR_BUFFER,
    invalid_count = MPI_ERR_COUNT,
    invalid_datatype = MPI_ERR_TYPE,
    invalid_tag = MPI_ERR_TAG,
    invalid_communicator = MPI_ERR_COMM,
    invalid_rank = MPI_ERR_RANK,
    invalid_request =MPI_ERR_REQUEST,
    invalid_root = MPI_ERR_ROOT,
    invalid_group = MPI_ERR_GROUP,
    invalid_operation = MPI_ERR_OP,
    invalid_topology = MPI_ERR_TOPOLOGY,
    invalid_dimension = MPI_ERR_DIMS,
    invalid_argument = MPI_ERR_ARG,
    unknown_error = MPI_ERR_UNKNOWN,
    message_truncated = MPI_ERR_TRUNCATE,
    other_error = MPI_ERR_OTHER,
    internal_error = MPI_ERR_INTERN,
    error_in_status = MPI_ERR_IN_STATUS,
    pending = MPI_ERR_PENDING,
    invalid_keyval = MPI_ERR_KEYVAL,
    not_enough_memory = MPI_ERR_NO_MEM,
    invalid_base = MPI_ERR_BASE,
    key_too_long = MPI_ERR_INFO_KEY,
    value_too_long = MPI_ERR_INFO_VALUE,
    invalid_key = MPI_ERR_INFO_NOKEY,
    spawn_error = MPI_ERR_SPAWN,
    invalid_port = MPI_ERR_PORT,
    invalid_service = MPI_ERR_SERVICE,
    invalid_name = MPI_ERR_NAME,
    invalid_win = MPI_ERR_WIN,
    invalid_size = MPI_ERR_SIZE,
    invalid_disp = MPI_ERR_DISP,
    invalid_info = MPI_ERR_INFO,
    invalid_locktype = MPI_ERR_LOCKTYPE,
    invalid_assert = MPI_ERR_ASSERT,
    rma_access_conflict = MPI_ERR_RMA_CONFLICT,
    rma_synchronization_error = MPI_ERR_RMA_SYNC,
#if MPI_VERSION >= 3
    rma_range_error = MPI_ERR_RMA_RANGE,
    rma_attach_failure = MPI_ERR_RMA_ATTACH,
    rma_share_failure = MPI_ERR_RMA_SHARED,
    rma_wrong_flavor = MPI_ERR_RMA_FLAVOR,
#endif
    invalid_file_handle = MPI_ERR_FILE,
    not_same = MPI_ERR_NOT_SAME,
    amode_error = MPI_ERR_AMODE,
    unsupported_datarep = MPI_ERR_UNSUPPORTED_DATAREP,
    unsupported_operation = MPI_ERR_UNSUPPORTED_OPERATION,
    no_such_file = MPI_ERR_NO_SUCH_FILE,
    file_exists = MPI_ERR_FILE_EXISTS,
    invalid_filename = MPI_ERR_BAD_FILE,
    permission_denied = MPI_ERR_ACCESS,
    no_space = MPI_ERR_NO_SPACE,
    quota_exceeded = MPI_ERR_QUOTA,
    read_only = MPI_ERR_READ_ONLY,
    file_in_use = MPI_ERR_FILE_IN_USE,
    duplicate_datarep = MPI_ERR_DUP_DATAREP,
    conversion_error = MPI_ERR_CONVERSION,
    other_io_error = MPI_ERR_IO,
};

} // namespace arb

namespace std {
template <> struct is_error_condition_enum<arb::mpi_errc>: true_type {};
}

namespace arb {

class mpi_error_category_impl;
ARB_ARBOR_API const mpi_error_category_impl& mpi_error_category();

class ARB_SYMBOL_VISIBLE mpi_error_category_impl: public std::error_category {
    const char* name() const noexcept override;
    std::string message(int) const override;
    std::error_condition default_error_condition(int) const noexcept override;
};

inline std::error_condition make_error_condition(mpi_errc ec) {
    return std::error_condition(static_cast<int>(ec), mpi_error_category());
}

struct ARB_SYMBOL_VISIBLE mpi_error: std::system_error {
    explicit mpi_error(int mpi_err):
        std::system_error(mpi_err, mpi_error_category()) {}

    mpi_error(int mpi_err, const std::string& what_arg):
        std::system_error(mpi_err, mpi_error_category(), what_arg) {}
};

} // namespace arb

