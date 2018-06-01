#pragma once

#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <domain_decomposition.hpp>
#include <recipe.hpp>
#include <util/unique_any.hpp>

namespace arb {

// Helper factory for building cell groups
cell_group_ptr cell_group_factory(const recipe& rec, const group_description& group);

} // namespace arb
