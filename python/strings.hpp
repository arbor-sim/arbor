#pragma once

/*
 * Utilities for generating string representations of types.
 */

#include <string>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>

#include "event_generator.hpp"

namespace pyarb {

std::string cell_member_string(const arb::cell_member_type&);
std::string context_string(const arb::context&);
std::string proc_allocation_string(const arb::proc_allocation&);
std::string schedule_explicit_string(const explicit_schedule_shim&);
std::string schedule_regular_string(const regular_schedule_shim&);
std::string schedule_poisson_string(const poisson_schedule_shim&);

} // namespace pyarb
