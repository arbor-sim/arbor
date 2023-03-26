#pragma once

#include <vector>

#include <arbor/domain_decomposition.hpp>
#include <arbor/network.hpp>
#include <arbor/recipe.hpp>

#include "connection.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"

namespace arb {

std::vector<connection> generate_network_connections(const recipe& rec,
    const context& ctx,
    const domain_decomposition& dom_dec);

} // namespace arb
