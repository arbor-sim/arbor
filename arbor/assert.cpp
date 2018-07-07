#include <iostream>

#include <arbor/assert.hpp>

#include "util/unwind.hpp"

namespace arb {

void abort_on_failed_assertion(
    const char* assertion,
    const char* file,
    int line,
    const char* func)
{
    // Emit stack trace If libunwind is being used.
    std::cerr << util::backtrace();

    // Explicit flush, as we can't assume default buffering semantics on stderr/cerr,
    // and abort() might not flush streams.
    std::cerr << file << ':' << line << " " << func
              << ": Assertion `" << assertion << "' failed." << std::endl;
    std::abort();
}

void ignore_failed_assertion(
    const char* assertion,
    const char* file,
    int line,
    const char* func)
{}

failed_assertion_handler_t global_failed_assertion_handler = abort_on_failed_assertion;

} // namespace arb
