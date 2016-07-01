#pragma once

namespace nest {
namespace mc {
namespace util {

bool failed_assertion(const char *assertion, const char *file, int line, const char *func);

} 
}
}


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

#endif //  def WITH_ASSERTIONS
