#pragma once

#include <iostream>

#include "array.hpp"
#include "definitions.hpp"
#include "host_coordinator.hpp"

#ifdef NMC_HAVE_CUDA
#include "device_coordinator.hpp"
#endif

namespace nest {
namespace mc {
namespace memory {

// specialization for host vectors
template <typename T>
using host_vector = array<T, host_coordinator<T>>;
template <typename T>
using host_view = array_view<T, host_coordinator<T>>;

template <typename T>
std::ostream& operator<< (std::ostream& o, host_view<T> const& v) {
    o << "[";
    for(auto const& value: v) o << value << ",";
    o << "]";

    return o;
}

#ifdef NMC_HAVE_CUDA
// specialization for pinned vectors. Use a host_coordinator, because memory is
// in the host memory space, and all of the helpers (copy, set, etc) are the
// same with and without page locked memory
template <typename T>
using pinned_vector = array<T, host_coordinator<T, pinned_allocator<T>>>;
template <typename T>
using pinned_view = array_view<T, host_coordinator<T, pinned_allocator<T>>>;

// specialization for device memory
template <typename T>
using device_vector = array<T, device_coordinator<T, cuda_allocator<T>>>;
template <typename T>
using device_view = array_view<T, device_coordinator<T, cuda_allocator<T>>>;
#endif

#ifdef WITH_KNL
// specialization for HBW memory on KNL
template <typename T>
using hwb_vector = array<T, host_coordinator<T, hwb_allocator<T>>>;
template <typename T>
using hwb_view = array_view<T, host_coordinator<T, hwb_allocator<T>>>;
#endif

} // namespace memory
} // namespace mc
} // namespace nest

// now import the helpers
// these require that host_vector etc have been defined
#include "wrappers.hpp"
#include "copy.hpp"
