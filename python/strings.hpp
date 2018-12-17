#pragma once

/*
 * Utilities for generating string representations of types.
 */

#include <string>

#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike.hpp>

namespace pyarb {

std::string cell_string(const arb::mc_cell&);
std::string cell_member_string(const arb::cell_member_type&);
std::string connection_string(const arb::cell_connection&);
std::string context_string(const arb::context&);
std::string group_description_string(const arb::group_description&);
std::string proc_allocation_string(const arb::proc_allocation&);
std::string proc_allocation_string(const arb::proc_allocation&);
std::string segment_location_string(const arb::segment_location&);
std::string spike_string(const arb::spike&);

} // namespace pyarb
