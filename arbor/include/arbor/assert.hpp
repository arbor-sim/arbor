#pragma once

#include <arbor/export.hpp>
#include <arbor/assert_macro.hpp>

namespace arb {

using failed_assertion_handler_t =
    void (*)(const char* assertion, const char* file, int line, const char* func);

ARB_ARBOR_API void abort_on_failed_assertion(const char* assertion, const char* file, int line, const char* func);
ARB_ARBOR_API void ignore_failed_assertion(const char* assertion, const char* file, int line, const char* func);

// defaults to abort_on_failed_assertion;
ARB_ARBOR_API extern failed_assertion_handler_t global_failed_assertion_handler;

} // namespace arb
