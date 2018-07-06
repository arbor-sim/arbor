#include <thread>

#ifdef ARB_HAVE_GPU
#include <cuda_runtime.h>
#endif

// TODO: C++17 use __has_include(<unistd.h>)
#if defined(__unix__) || defined(__APPLE__) && defined(__MACH__)
#include <unistd.h>
#endif

#include "affinity.hpp"
#include "node_info.hpp"

namespace arb {
namespace hw {

unsigned node_gpus() {
#ifdef ARB_HAVE_GPU
    int n;
    if (!cudaGetDeviceCount(&n)) {
        return static_cast<unsigned>(n);
    }
#endif

    return 0;
}

unsigned node_processors() {
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

    return n;
}

} // namespace util
} // namespace arb
