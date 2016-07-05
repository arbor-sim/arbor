#pragma once

#include <iostream>
#include <sstream>
#include <mutex>

namespace nest {
namespace mc {
namespace util {

bool failed_assertion(const char* assertion, const char* file, int line, const char* func);
std::ostream &debug_emit_trace_leader(std::ostream& out, const char* file, int line, const char* varlist);

inline void debug_emit(std::ostream& out) {
    out << "\n";
}

template <typename Head, typename... Tail>
void debug_emit(std::ostream& out, const Head& head, const Tail&... tail) {
    out << head;
    if (sizeof...(tail)) {
        out << ", ";
    }
    debug_emit(out, tail...);
}

#ifdef WITH_TBB
extern std::mutex global_debug_cerr_mutex;
#endif

template <typename... Args>
void debug_emit_trace(const char* file, int line, const char* varlist, const Args&... args) {
#ifdef WITH_TBB
    std::stringstream out;
#else
    auto &out = std::cerr;
#endif

    debug_emit_trace_leader(out, file, line, varlist);
    debug_emit(out, args...);

#ifdef WITH_TBB
    std::lock_guard<std::mutex> _(global_debug_cerr_mutex);
    std::cerr << out.rdbuf();
#else
    out.flush();
#endif
}

} // namespace util
} // namespace mc
} // namespace nest

#ifdef WITH_TRACE
    #define TRACE(vars...) nest::mc::util::debug_emit_trace(__FILE__, __LINE__, #vars, ##vars)
#else
    #define TRACE(...)
#endif


#ifdef WITH_ASSERTIONS
    #ifdef __GNUC__
        #define DEBUG_FUNCTION_NAME __PRETTY_FUNCTION__
    #else
        #define DEBUG_FUNCTION_NAME __func__
    #endif

    #define EXPECTS(condition) \
       (void)((condition) || \
       nest::mc::util::failed_assertion(#condition, __FILE__, __LINE__, DEBUG_FUNCTION_NAME))
#else
    #define EXPECTS(condition)
#endif // def WITH_ASSERTIONS
