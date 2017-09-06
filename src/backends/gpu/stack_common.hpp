#pragma once

#include <backends/fvm_types.hpp>

namespace nest {
namespace mc {
namespace gpu {

/// stores a single crossing event
struct threshold_crossing {
    fvm_size_type index;    // index of variable
    fvm_value_type time;    // time of crossing

    friend bool operator==(threshold_crossing l, threshold_crossing r) {
        return l.index==r.index && l.time==r.time;
    }
};

template <typename T>
struct stack_base {
    using value_type = T;

    // The number of items of type value_type that can be stored in the stack
    unsigned capacity;

    // The number of items that have been stored
    unsigned size;

    // Memory containing the value buffer
    value_type* data;
};


} // namespace gpu
} // namespace mc
} // namespace nest
