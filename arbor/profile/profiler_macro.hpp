#pragma once

#include <arbor/profile/profiler.hpp>

#ifdef ARB_GPU_ENABLED
#include <arbor/gpu/gpu_api.hpp>
#else
namespace arb::gpu {
inline void profile_push(const char* str) {}
inline void profile_pop() {}
}
#endif

#ifdef ARB_HAVE_PROFILING

    // enter a profiling region
#define PE(name)                                                                     \
        {                                                                            \
            static std::size_t region_id_ = arb::profile::profiler_region_id(#name); \
            arb::gpu::profile_push(#name);                                           \
            arb::profile::profiler_enter(region_id_);                                \
        }

    // leave a profling region
#define PL(name)                                                                     \
        {                                                                            \
            static std::size_t region_id_ = arb::profile::profiler_region_id(#name); \
            arb::gpu::profile_pop();                                                 \
            arb::profile::profiler_leave(region_id_);                                \
        }

#else

#define PE(name)
#define PL(name)

#endif

