#include <cstdlib>
#include <exception>
#include <regex>
#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/util/optional.hpp>
#include <hardware/node_info.hpp>

#include "threading.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace threading {

// Test environment variables for user-specified count of threads.
//
// ARB_NUM_THREADS is used if set, otherwise OMP_NUM_THREADS is used.
//
// If neither variable is set, returns no value.
//
// Valid values for the environment variable are:
//  0 : Arbor is responsible for picking the number of threads.
//  >0: The number of threads to use.
//
// Throws std::runtime_error:
//  ARB_NUM_THREADS or OMP_NUM_THREADS is set with invalid value.
util::optional<size_t> get_env_num_threads() {
    const char* str;

    // select variable to use:
    //   If ARB_NUM_THREADS_VAR is set, use $ARB_NUM_THREADS_VAR
    //   else if ARB_NUM_THREAD set, use it
    //   else if OMP_NUM_THREADS set, use it
    if (auto nthreads_var_name = std::getenv("ARB_NUM_THREADS_VAR")) {
        str = std::getenv(nthreads_var_name);
    }
    else if (! (str = std::getenv("ARB_NUM_THREADS"))) {
        str = std::getenv("OMP_NUM_THREADS");
    }

    // If the selected var is unset set the number of threads to
    // the hint given by the standard library
    if (!str) {
        return util::nullopt;
    }

    auto nthreads = std::strtoul(str, nullptr, 10);

    // check that the environment variable string describes a non-negative integer
    if (errno==ERANGE ||
        !std::regex_match(str, std::regex("\\s*\\d*[0-9]\\d*\\s*")))
    {
        throw arbor_exception(util::pprintf(
            "requested number of threads \"{}\" is not a valid value", str));
    }

    return nthreads;
}

std::size_t num_threads_init() {
    std::size_t n = 0;

    if (auto env_threads = get_env_num_threads()) {
        n = env_threads.value();
    }

    if (!n) {
        n = hw::node_processors();
    }

    return n? n: 1;
}

// Returns the number of threads used by the threading back end.
// Throws:
//      std::runtime_error if an invalid environment variable was set for the
//      number of threads.
size_t num_threads() {
    // TODO: this is a bit of a hack until we have user-configurable threading.
#if defined(ARB_HAVE_SERIAL)
    return 1;
#else
    static size_t num_threads_cached = num_threads_init();
    return num_threads_cached;
#endif
}

} // namespace threading
} // namespace arb
