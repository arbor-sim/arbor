#include <cstdio>
#include <cstdlib>

#include "util/debug.hpp"

bool nest::mc::util::failed_assertion(const char *assertion, const char *file,
                                      int line, const char *func)
{
    std::fprintf(stderr, "%s:%d %s: Assertion `%s' failed.\n", file, line, func, assertion);
    std::abort();
    return false;
}
