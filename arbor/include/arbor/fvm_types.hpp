#pragma once

#include <arbor/common_types.hpp>

// Basic types shared across FVM implementations/backends.

namespace arb {

using fvm_value_type = double;
using fvm_size_type = cell_local_size_type;
using fvm_index_type = int;

struct fvm_gap_junction {
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;

    std::pair<index_type, index_type> loc;
    value_type weight;

    fvm_gap_junction() {}
    fvm_gap_junction(std::pair<index_type, index_type> l, value_type w):
        loc(l),
        weight(w) {}
};

// TODO move me somewhere more sensible
struct ion_state_view {
    fvm_value_type* current_density;
    fvm_value_type* reversal_potential;
    fvm_value_type* internal_concentration;
    fvm_value_type* external_concentration;
    fvm_value_type* ionic_charge;
};

} // namespace arb
