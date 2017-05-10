#pragma once

#include <vector>

#include <backends.hpp>
#include <cell_group.hpp>
#include <util/unique_any.hpp>

namespace nest {
namespace mc {

// Helper factory for building cell groups
cell_group_ptr cell_group_factory(
    cell_kind kind,
    cell_gid_type first_gid,
    const std::vector<util::unique_any>& cells,
    backend_policy backend);

} // namespace mc
} // namespace nest
