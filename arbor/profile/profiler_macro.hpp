#pragma once

#include <string>

#ifdef ARB_HAVE_PROFILING
#include <tracy/Tracy.hpp>
#endif


#ifdef ARB_HAVE_PROFILING
#ifdef ARB_HAVE_STACK_PROFILING // include stacks
    #define PROFILE_ZONE() ZoneScopedS(16)
    #define PROFILE_NAMED_ZONE(name) ZoneScopedNS(name, 16)
#else // just plain profile
    #define PROFILE_ZONE() ZoneScoped
    #define PROFILE_NAMED_ZONE(name) ZoneScopedN(name)
#endif
   #define ANNOTATE_ZONE(tag, len) ZoneText(tag, len)
   #define PROFILE_END_EPOCH() FrameMark
#else // No profiling
    #define PROFILE_ZONE()
    #define PROFILE_NAMED_ZONE(name)
    #define ANNOTATE_ZONE(tag, len)
    #define PROFILE_END_EPOCH()
#endif

#ifdef ARB_HAVE_MEMORY_PROFILING
    #define MARK_ALLOC(ptr, len) TracyAlloc(ptr, len)
    #define MARK_FREE(ptr) TracyFree(ptr)
#else
    #define MARK_ALLOC(ptr, len)
    #define MARK_FREE(ptr)
#endif


inline
void DECLARE_THREAD(const std::string& tag) {
#ifdef ARB_HAVE_PROFILING
    tracy::SetThreadName(tag.c_str());
#endif
}
