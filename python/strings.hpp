#pragma once

/*
 * Utilities for generating string representations of types.
 */

#include <string>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

namespace arb {
namespace py {

std::string cell_member_string(const arb::cell_member_type&);
std::string local_resources_string(const arb::local_resources&);
std::string proc_allocation_string(const arb::proc_allocation&);
std::string context_string(const arb::context&);

} // namespace py
} // namespace arb
