#pragma once

#include <arbor/common_types.hpp>

// Basic types shared across FVM implementations/backends.

namespace arb {

using fvm_value_type = double;
using fvm_size_type = cell_local_size_type;
using fvm_index_type = int;

} // namespace arb
