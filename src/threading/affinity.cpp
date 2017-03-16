#include <vector>

#include <cstdlib>

#ifdef __linux__

    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif

    extern "C" {
        #include <sched.h>
    }

#endif

namespace nest {
namespace mc {
namespace threading {

#ifdef __linux__
std::vector<int> get_affinity() {
    cpu_set_t cpu_set_mask;

    auto status = sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set_mask);

    if(status==-1) {
        return {};
    }

    auto cpu_count = CPU_COUNT(&cpu_set_mask);

    std::vector<int> cores;
    for(auto i=0; i<CPU_SETSIZE && cores.size()<cpu_count; ++i) {
        if(CPU_ISSET(i, &cpu_set_mask)) {
            cores.push_back(i);
        }
    }

    if(cores.size() != cpu_count) {
        return {};
    }

    return cores;
}
#else

// No support for non-linux systems
std::vector<int> get_affinity() {
    return {};
}
#endif

unsigned count_available_cores() {
    auto n = get_affinity().size();

    // Assume that there must be at least 1 core if an error was encountered.
    return n? n: 1;
}

} // namespace threading
} // namespace mc
} // namespace nest
