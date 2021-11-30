#pragma once

#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {

// Concrete storage of gpu stack datatype.
// The stack datatype resides in host memory, and holds a pointer to the
// stack_storage in device memory.
template <typename T>
struct stack_storage {
    using value_type = T;

    // The number of items of type value_type that can be stored in the stack
    unsigned capacity;

    // The number of items that have been stored.
    // This may exceed capacity if more stores were attempted than it is
    // possible to store, in which case only the first capacity values are valid.
    unsigned stores;

    // Memory containing the value buffer
    value_type* data;
};


} // namespace gpu
} // namespace arb
