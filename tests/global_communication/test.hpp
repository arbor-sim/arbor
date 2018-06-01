#pragma once

#include <communication/distributed_context.hpp>

// Global context is a global variable, set in the main() funtion of the main
// test driver test.cpp.
extern arb::distributed_context g_context;
