#pragma once

#include <arbor/domain_decomposition.hpp>
#include <arbor/recipe.hpp>

#include "cell_group.hpp"

namespace arb {

// Helper factory for building cell groups
cell_group_ptr cell_group_factory(const recipe& rec, const group_description& group);

} // namespace arb
