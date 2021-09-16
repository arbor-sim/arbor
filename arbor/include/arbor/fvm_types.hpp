#pragma once

#include <arbor/arb_types.h>
#include <arbor/common_types.hpp>

namespace arb {

using fvm_value_type = arb_value_type;
using fvm_size_type  = arb_size_type;
using fvm_index_type = arb_index_type;

struct fvm_gap_junction {
    cell_lid_type local_idx;
    fvm_index_type local_cv;
    fvm_index_type peer_cv;
    arb_value_type weight;
};
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(fvm_gap_junction, (a.local_cv, a.peer_cv, a.local_idx, a.weight), (b.local_cv, b.peer_cv, b.local_idx, b.weight))

} // namespace arb
