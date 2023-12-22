#pragma once

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

// Generate and return list of connections from the network description of the recipe.
// Does not include connections from the "connections_on" recipe function.
// Only returns connections with local cell targets as described in the domain decomposition.
ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec);

// Generate and return list of ALL connections from the network description of the recipe.
// Does not include connections from the "connections_on" recipe function.
ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec);

}  // namespace arb
