#pragma once

#include "../stack_common.hpp"

namespace nest {
namespace mc {
namespace gpu {

template <typename T>
__device__
void push_back(stack_base<T>& s, const T& value) {
    // Atomically increment the size counter. The atomicAdd returns
    // the value of size before the increment, which is the location
    // at which this thread can store value.
    unsigned position = atomicAdd(&(s.size), 1u);

    // It is possible that size>capacity. In this case, only capacity
    // entries are stored, and additional values are lost. The size
    // contains the total number of attempts to push.
    if (position<s.capacity) {
        s.data[position] = value;
    }
}

} // namespace gpu
} // namespace mc
} // namespace nest
