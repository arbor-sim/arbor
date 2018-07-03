#pragma once

#include <vector>

#include <arbor/util/unique_any.hpp>

#include "backends.hpp"
#include "cell_group.hpp"
#include "domain_decomposition.hpp"
#include "recipe.hpp"

namespace arb {

// Helper factory for building cell groups
cell_group_ptr cell_group_factory(const recipe& rec, const group_description& group);

} // namespace arb
