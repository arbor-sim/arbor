#pragma once

#include <arbor/arb_types.hpp>
#include <arbor/common_types.hpp>

namespace arb {

struct fvm_gap_junction {
    cell_lid_type local_idx; // Index relative to other gap junction sites on the cell.
    arb_size_type local_cv;  // CV index of the local gap junction site.
    arb_size_type peer_cv;   // CV index of the peer gap junction site.
    arb_value_type weight;   // unit-less local weight of the connection.
};
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(fvm_gap_junction, (a.local_cv, a.peer_cv, a.local_idx, a.weight), (b.local_cv, b.peer_cv, b.local_idx, b.weight))

} // namespace arb
