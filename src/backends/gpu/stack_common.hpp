#pragma once

#include "backends/fvm_types.hpp"

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

    __device__
    void push_back(const value_type& value) {
        // Atomically increment the size_ counter. The atomicAdd returns
        // the value of size_ before the increment, which is the location
        // at which this thread can store value.
        unsigned position = atomicAdd(&size, 1u);

        // It is possible that size_>capacity_. In this case, only capacity_
        // entries are stored, and additional values are lost. The size_
        // will contain the total number of attempts to push,
        if (position<capacity) {
            data[position] = value;
        }
    }
};


} // namespace gpu
} // namespace mc
} // namespace nest
