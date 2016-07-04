#include <cstdlib>
#include <iostream>

#include "util/debug.hpp"

bool nest::mc::util::failed_assertion(const char *assertion, const char *file,
                                      int line, const char *func)
{
    // Explicit flush, as we can't assume default buffering semantics on stderr/cerr,
    // and abort() might not flush streams.

    std::cerr << file << ':' << line << " " << func
              << ": Assertion `" << assertion << "' failed." << std::endl;
    std::abort();
    return false;
}
