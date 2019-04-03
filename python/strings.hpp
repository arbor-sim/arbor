#pragma once

#include <string>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

// Utilities for generating string representations of types.
namespace pyarb {

std::string cell_member_string(const arb::cell_member_type&);
std::string context_string(const arb::context&);
std::string proc_allocation_string(const arb::proc_allocation&);

} // namespace pyarb
