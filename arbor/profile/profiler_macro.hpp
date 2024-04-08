#pragma once

#include <arbor/profile/profiler.hpp>

#ifdef ARB_HAVE_PROFILING

    #define DECLARE_THREAD(id) tracy::SetThreadName("thread" #id)

#ifdef ARB_HAVE_STACK_PROFILING
    #define PROFILE_ZONE(name) ZoneScopedNS(name, 16)
#else
    #define PROFILE_ZONE(name) ZoneScopedN(name)
#endif

#else
    #define DECLARE_THREAD(id)
    #define PROFILE_ZONE(name)
#endif
