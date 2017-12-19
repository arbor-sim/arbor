#include <cstdlib>
#include <vector>

#include <util/optional.hpp>

#ifdef __linux__

    #ifndef _GNU_SOURCE
        #define _GNU_SOURCE
    #endif

    extern "C" {
        #include <sched.h>
    }

#endif

namespace arb {
namespace hw {

#ifdef __linux__
std::vector<int> get_affinity() {
    cpu_set_t cpu_set_mask;

    auto status = sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set_mask);

    if(status==-1) {
        return {};
    }

    unsigned cpu_count = CPU_COUNT(&cpu_set_mask);

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

util::optional<std::size_t> num_cores() {
    auto cores = get_affinity();
    if (cores.size()==0u) {
        return util::nullopt;
    }
    return cores.size();
}

} // namespace hw
} // namespace arb
