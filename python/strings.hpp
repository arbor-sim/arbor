#pragma once

/*
 * Utilities for generating string representations of types.
 */

#include <string>

#include <arbor/context.hpp>

namespace pyarb {

std::string context_string(const arb::context&);
std::string proc_allocation_string(const arb::proc_allocation&);

} // namespace pyarb
