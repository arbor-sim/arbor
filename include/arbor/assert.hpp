#pragma once

#include <arbor/assert_macro.hpp>

namespace arb {

using failed_assertion_handler_t =
    void (*)(const char* assertion, const char* file, int line, const char* func);

void abort_on_failed_assertion(const char* assertion, const char* file, int line, const char* func);
void ignore_failed_assertion(const char* assertion, const char* file, int line, const char* func);

// defaults to abort_on_failed_assertion;
extern failed_assertion_handler_t global_failed_assertion_handler;

} // namespace arb
