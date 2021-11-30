#pragma once

#include <arbor/arb_types.h>
#include <arbor/common_types.hpp>

namespace arb {

using fvm_value_type = arb_value_type;
using fvm_size_type  = arb_size_type;
using fvm_index_type = arb_index_type;

struct fvm_gap_junction {
    cell_lid_type local_idx; // Index relative to other gap junction sites on the cell.
    fvm_size_type local_cv;  // CV index of the local gap junction site.
    fvm_size_type peer_cv;   // CV index of the peer gap junction site.
    fvm_value_type weight;   // unit-less local weight of the connection.
};
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(fvm_gap_junction, (a.local_cv, a.peer_cv, a.local_idx, a.weight), (b.local_cv, b.peer_cv, b.local_idx, b.weight))

} // namespace arb
