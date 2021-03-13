#pragma once
#include "printeropt.hpp"

extern bool options_trace_codegen;

#define ENTERM(stream, msg) do {                                        \
        if (options_trace_codegen) {                                    \
            (stream) << " /* " << __FUNCTION__ << ":" << (msg) << ":enter */ "; \
        }                                                               \
    } while(0)

#define EXITM(stream, msg) do {                                         \
        if (options_trace_codegen) {                                    \
            (stream) << " /* " << __FUNCTION__ << ":" << (msg) << ":exit */ "; \
        }                                                               \
    } while(0)

#define ENTER(stream) ENTERM(stream, "")
#define EXIT(stream) EXITM(stream, "")
