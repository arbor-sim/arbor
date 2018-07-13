#pragma once

#include <arbor/execution_context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

#include "hardware/node_info.hpp"

namespace arb {

domain_decomposition partition_load_balance(const recipe& rec,
                                            hw::node_info nd,
                                            const execution_context* ctx);

} // namespace arb
