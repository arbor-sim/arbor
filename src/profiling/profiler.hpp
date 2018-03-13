#pragma once

#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <vector>

#include <threading/threading.hpp>

namespace arb {
namespace util {

// type used for region identifiers
using region_id_type = std::size_t;

// The results of a profiler run.
struct profile {
    // the name of each profiled region.
    std::vector<std::string> names;

    // the number of times each region was called.
    std::vector<std::size_t> counts;

    // the accumulated time spent in each region.
    std::vector<double> times;

    // the number of threads for which profiling information was recorded.
    std::size_t num_threads;

    // the wall time between profile_start() and profile_stop().
    double wall_time;
};

void profiler_clear();
void profiler_enter(std::size_t region_id);
void profiler_leave();

profile profiler_summary();
std::size_t profiler_region_id(const char* name);

std::ostream& operator<<(std::ostream&, const profile&);

#ifdef ARB_HAVE_PROFILING

    // enter a profiling region
    #define REGION_TAG_NAME(x) x##_profile_region_tag__
    #define PE(name) \
        { \
            static std::size_t REGION_TAG_NAME(name) = arb::util::profiler_region_id(#name); \
            arb::util::profiler_enter(REGION_TAG_NAME(name)); \
        }

    // leave a profling region
    #define PL arb::util::profiler_leave

#else

    #define PE(name)
    #define PL()

#endif

} // namespace util
} // namespace arb

