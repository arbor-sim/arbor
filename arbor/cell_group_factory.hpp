#pragma once

// Provide a map from cell group kinds and execution back-end to implementation,
// as represented by a `cell_group_factory` function wrapper below.
//
// An empty function implies there is no support for that cell kind on that
// back-end.

#include <functional>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/recipe.hpp>

#include "cell_group.hpp"
#include "execution_context.hpp"

namespace arb {

using cell_group_factory = std::function<
        cell_group_ptr(const std::vector<cell_gid_type>&, const recipe&)>;

cell_group_factory cell_kind_implementation(
        cell_kind, backend_kind, const execution_context&);

inline bool cell_kind_supported(
        cell_kind c, backend_kind b, const execution_context& ctx)
{
    return static_cast<bool>(cell_kind_implementation(c, b, ctx));
}

} // namespace arb
