#include <cstdlib>
#include <system_error>
#include <vector>

#include <arborenv/concurrency.hpp>

#ifdef __linux__

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

extern "C" {
#include <sched.h>
}

namespace arbenv {

ARB_ARBORENV_API std::vector<int> get_affinity() {
    std::vector<int> cores;
    cpu_set_t cpu_set_mask;

    int status = sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set_mask);
    if (status) {
        throw std::system_error(errno, std::generic_category());
    }

    for (int i=0; i<CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &cpu_set_mask)) {
            cores.push_back(i);
        }
    }

    return cores;
}

} // namespace arbenv

#else // def __linux__

// No support for non-linux systems.
namespace arbenv {

ARB_ARBORENV_API std::vector<int> get_affinity() {
    return {};
}

} // namespace arbenv

#endif // def __linux__
