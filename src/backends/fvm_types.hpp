#pragma once

#include <common_types.hpp>

// Basic types shared across FVM implementations/backends.

namespace nest {
namespace mc {

using fvm_value_type = double;
using fvm_size_type = cell_local_size_type;

} // namespace mc
} // namespace nest
