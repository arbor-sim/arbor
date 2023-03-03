#pragma once

#include <vector>

#include <arbor/domain_decomposition.hpp>
#include <arbor/network.hpp>
#include <arbor/recipe.hpp>

#include "connection.hpp"
#include "distributed_context.hpp"
#include "label_resolution.hpp"

namespace arb {

std::vector<connection> generate_network_connections(
    const std::vector<network_description>& descriptions,
    const connectivity& rec,
    const distributed_context& distributed,
    const domain_decomposition& dom_dec,
    const label_resolution_map& source_resolution_map,
    const label_resolution_map& target_resolution_map);

} // namespace arb
