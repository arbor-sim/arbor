#pragma once

// Use heuristics, environment variables to determine a suitable context
// proc_allocation.

#include <arbor/context.hpp>

#include <arborenv/arbenvexcept.hpp>
#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>
#include <arborenv/export.hpp>

namespace arbenv {

// Best-effort heuristics for thread utilization: use ARBENV_NUM_THREADS value
// if set and non-zero, throwing arbev::invalid_env_value if it has an invalid
// value, or else return the value determined by arbenv::thread_concurrency().

ARB_ARBORENV_API unsigned long default_concurrency();

// If Arbor is built without GPU support, return -1.
//
// If the ARBENV_GPU_ID environment variable is set, return -1 if it is less
// than zero (indicating no GPU should be used), or its integer value if it is
// a valid GPU device id.
//
// If ARBENV_GPU_ID is not set or empty, return 0 if 0 is a valid GPU device
// id, and -1 otherwise.
//
// Throws arbenv::invalid_env_value if ARBENV_GPU_ID is not an int value, or
// arbenv::no_such_gpu if it doesn't correspond to a valid GPU id.

ARB_ARBORENV_API int default_gpu();

// Construct default proc_allocation from `default_concurrency()` and
// `default_gpu()`.

inline arb::proc_allocation default_allocation() {
    return arb::proc_allocation{static_cast<unsigned>(default_concurrency()), default_gpu()};
}

// Retrieve user-specified thread count from ARBENV_NUM_THREADS environment variable.
//
// * Throws arbenv::invalid_env_value if ARBENV_NUM_THREADS is set but contains a
//   non-numeric, non-positive, or out of range value.
// * Returns zero if ARBENV_NUM_THREADS is unset, or set and empty.

ARB_ARBORENV_API unsigned long get_env_num_threads();

} // namespace arbenv
