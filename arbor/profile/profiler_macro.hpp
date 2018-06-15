#pragma once

#include <arbor/profile/profiler.hpp>

#ifdef ARB_HAVE_PROFILING

    // enter a profiling region
    #define PE(name) \
        { \
            static std::size_t region_id__ = arb::profile::profiler_region_id(#name); \
            arb::profile::profiler_enter(region_id__); \
        }

    // leave a profling region
    #define PL arb::profile::profiler_leave

#else

    #define PE(name)
    #define PL()

#endif

