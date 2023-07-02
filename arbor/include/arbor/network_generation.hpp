#pragma once

#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

ARB_ARBOR_API std::vector<network_connection_info> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec);

}  // namespace arb
