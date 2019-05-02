#include <cstdlib>
#include <regex>
#include <string>
#include <thread>

#include <arborenv/concurrency.hpp>

// TODO: C++17 use __has_include(<unistd.h>)
#if defined(__unix__) || defined(__APPLE__) && defined(__MACH__)
#include <unistd.h>
#endif

namespace arbenv {

// Test environment variables for user-specified count of threads.
unsigned get_env_num_threads() {
    using namespace std::literals;
    const char* str;

    // select variable to use:
    //   If ARB_NUM_THREADS_VAR is set, use $ARB_NUM_THREADS_VAR
    //   else if ARB_NUM_THREADS set, use it
    //   else if OMP_NUM_THREADS set, use it
    if (auto nthreads_var_name = std::getenv("ARB_NUM_THREADS_VAR")) {
        str = std::getenv(nthreads_var_name);
    }
    else if (! (str = std::getenv("ARB_NUM_THREADS"))) {
        str = std::getenv("OMP_NUM_THREADS");
    }

    // No environment variable set, so return 0.
    if (!str) {
        return 0;
    }

    errno = 0;
    auto nthreads = std::strtoul(str, nullptr, 10);

    // check that the environment variable string describes a non-negative integer
    if (errno==ERANGE ||
        !std::regex_match(str, std::regex("\\s*\\d*[0-9]\\d*\\s*")))
    {
        errno = 0;
        throw std::runtime_error("Requested number of threads \""s + str + "\" is not a valid value"s);
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
    n = n? n: 1;

    return n;
}

} // namespace arbenv
