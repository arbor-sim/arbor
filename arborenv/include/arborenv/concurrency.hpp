#pragma once

#include <vector>

namespace arbenv {

// Test environment variables for user-specified count of threads.
// Potential environment variables are tested in this order:
//   1. use the environment variable specified by ARB_NUM_THREADS_VAR
//   2. use ARB_NUM_THREADS
//   3. use OMP_NUM_THREADS
//
// Valid values for the environment variable are:
//      0 : Arbor is responsible for picking the number of threads.
//     >0 : The number of threads to use.
//
// Returns:
//   >0 : the number of threads set by environment variable.
//    0 : value is not set in environment variable.
//
// Throws std::runtime_error:
//      Environment variable is set with invalid value.
unsigned get_env_num_threads();

// Take a best guess at the number of threads that can be run concurrently.
// Will return at least 1.
unsigned thread_concurrency();

// The list of logical processors for which the calling thread has affinity.
// If calling from the main thread at application start up, before
// attempting to change thread affinity, may produce unreliable
// results.
//  - beware thread pinning or custom job scheduler affinity
//    flags that assign threads to specific cores.
//
// Returns an empty vector if unable to determine the number of
// available cores.
std::vector<int> get_affinity();

} // namespace arbenv
