#pragma once

#include <arbor/fvm_types.hpp>

namespace arb {

// Representation of a single crossing event.

struct threshold_crossing {
    arb_size_type index;    // index of variable
    arb_value_type time;    // time of crossing

    friend bool operator==(threshold_crossing l, threshold_crossing r) {
        return l.index==r.index && l.time==r.time;
    }
};

} // namespace arb
