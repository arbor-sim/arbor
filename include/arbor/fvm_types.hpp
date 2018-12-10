#pragma once

#include <arbor/common_types.hpp>

// Basic types shared across FVM implementations/backends.

namespace arb {

using fvm_value_type = double;
using fvm_size_type = cell_local_size_type;
using fvm_index_type = int;

struct gap_junction {
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;

    std::pair<index_type, index_type> loc;
    std::pair<value_type, value_type> area;
    value_type ggap; //μS

    gap_junction() {}
    gap_junction(std::pair<index_type, index_type> l, std::pair<value_type, value_type> a,
                 value_type g): loc(l), area(a), ggap(g) {}

};

} // namespace arb
