#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

#include <arborenv/concurrency.hpp>

#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

namespace arbenv {

// Take a best guess at the number of threads that can be run concurrently.
// Will return at least 1.
ARB_ARBORENV_API unsigned long thread_concurrency() {
    // Attempt to get count first from affinity information if available.
    unsigned long n = get_affinity().size();

    // If no luck, try sysconf.
#ifdef _SC_NPROCESSORS_ONLN
    if (!n) {
        long r = sysconf(_SC_NPROCESSORS_ONLN);
        if (r>0) {
            n = (unsigned long)r;
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
