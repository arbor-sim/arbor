#pragma once

#include <arbor/profile/profiler.hpp>

#ifdef ARB_HAVE_PROFILING

    // initialize profiler
    #define PI(ts) arb::profile::profiler_initialize(ts)

    // enter a profiling region
    #define PE(name) \
        { \
            static std::size_t region_id_ = arb::profile::profiler_region_id(#name); \
            arb::profile::profiler_enter(region_id_); \
        }

    // leave a profling region
    #define PL arb::profile::profiler_leave

#else

    #define PI(ts)
    #define PE(name)
    #define PL()

#endif
