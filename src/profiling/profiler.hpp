#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <threading/threading.hpp>

namespace arb {
namespace util {

using timer_type = arb::threading::timer;
using time_point = timer_type::time_point;

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

void profiler_start();
void profiler_stop();
void profiler_restart();
void profiler_enter(std::size_t region_id);
void profiler_leave();

profile profiler_summary();
std::size_t profiler_region_id(const char* name);
void profiler_print(const profile& prof, float threshold=1);

#define PL arb::util::profiler_leave
#define PE(name) \
    static std::size_t name ## _profile_region_tag__ = arb::util::profiler_region_id(#name); \
    arb::util::profiler_enter(name ## _profile_region_tag__);

} // namespace util
} // namespace arb
