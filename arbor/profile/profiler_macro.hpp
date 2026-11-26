#pragma once

#include <arbor/profile/profiler.hpp>

#ifdef ARB_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace arb::gpu {
// With NVTX, the profiler regions also show up as ranges in NVIDIA's tools
// (Nsight Systems). NVTX is header-only and only used if found by CMake.
inline void profile_push(const char* str) {
#ifdef ARB_HAVE_NVTX
    nvtxRangePushA(str);
#endif
}

inline void profile_pop() {
#ifdef ARB_HAVE_NVTX
    nvtxRangePop();
#endif
}
}

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

