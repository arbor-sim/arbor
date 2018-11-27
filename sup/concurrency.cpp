#include <algorithm>
#include <cstdlib>
#include <exception>
#include <regex>
#include <string>
#include <thread>

#include <arbor/arbexcept.hpp>
#include <arbor/util/optional.hpp>

#include <sup/affinity.hpp>
#include <sup/concurrency.hpp>

// TODO: C++17 use __has_include(<unistd.h>)
#if defined(__unix__) || defined(__APPLE__) && defined(__MACH__)
#include <unistd.h>
#endif

namespace sup {

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
arb::util::optional<size_t> get_env_num_threads() {
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
        return arb::util::nullopt;
    }

    errno = 0;
    auto nthreads = std::strtoul(str, nullptr, 10);

    // check that the environment variable string describes a non-negative integer
    if (errno==ERANGE ||
        !std::regex_match(str, std::regex("\\s*\\d*[0-9]\\d*\\s*")))
    {
        throw arb::arbor_exception(
            std::string("requested number of threads \"") + str + "\" is not a valid value");
    }
    errno = 0;

    return nthreads;
}

// Take a best guess at the number of threads that can be run concurrently.
// Will return at least 1.
unsigned thread_concurrency() {
    // Attempt to get count first from affinity information if available.
    unsigned n = get_affinity().size();

    // If no luck, try sysconf.
#ifdef _SC_NPROCESSORS_ONLN
    if (!n) {
        long r = sysconf(_SC_NPROCESSORS_ONLN);
        if (r>0) {
            n = (unsigned)r;
        }
    }
#endif

    // If still zero, try the hint from the library.
    if (!n) {
        n = std::thread::hardware_concurrency();
    }

    // If still zero, use one thread.
    n = std::max(n, 1u);

    return n;
}

} // namespace sup
