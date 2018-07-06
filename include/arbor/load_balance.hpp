#pragma once

#include <arbor/distributed_context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

namespace arb {

domain_decomposition partition_load_balance(
    const recipe& rec, domain_info nd, const distributed_context* ctx);

} // namespace arb
