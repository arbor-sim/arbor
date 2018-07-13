#pragma once

#include <arbor/util/optional.hpp>

namespace arb {
namespace threading {

// Test environment variables for user-specified count of threads.
// Potential environment variables are tested in this order:
//   1. use the environment variable specified by ARB_NUM_THREADS_VAR
//   2. use ARB_NUM_THREADS
//   3. use OMP_NUM_THREADS
//   4. If no variable is set, returns no value.
//
// Valid values for the environment variable are:
//      0 : Arbor is responsible for picking the number of threads.
//     >0 : The number of threads to use.
//
// Throws std::runtime_error:
//      Environment variable is set with invalid value.
util::optional<size_t> get_env_num_threads();

size_t num_threads();

} // namespace threading
} // namespace arb

//#if defined(ARB_HAVE_TBB)

//#include "tbb.hpp"

//#elif defined(ARB_HAVE_CTHREAD)

#include "cthread.hpp"

//#else

//#define ARB_HAVE_SERIAL
//#include "serial.hpp"

//#endif
