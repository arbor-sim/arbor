#pragma once

#include <common_types.hpp>

// Basic types shared across FVM implementations/backends.

namespace arb {

using fvm_value_type = double;
using fvm_size_type = cell_local_size_type;
using fvm_index_type = int;

// Stores a single crossing event.

struct threshold_crossing {
    fvm_size_type index;    // index of variable
    fvm_value_type time;    // time of crossing

    friend bool operator==(threshold_crossing l, threshold_crossing r) {
        return l.index==r.index && l.time==r.time;
    }
};

} // namespace arb
