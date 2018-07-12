#pragma once

#include <arbor/distributed_context.hpp>
#include <arbor/execution_context.hpp>

// Global context is a global variable, set in the main() funtion of the main
// test driver test.cpp.
extern arb::execution_context g_context;
