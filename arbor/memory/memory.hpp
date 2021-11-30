#pragma once

#include <iostream>

#include "array.hpp"
#include "definitions.hpp"
#include "host_coordinator.hpp"
#include "device_coordinator.hpp"

namespace arb {
namespace memory {

// specialization for host vectors
template <typename T>
using host_vector = array<T, host_coordinator<T>>;
template <typename T>
using host_view = array_view<T, host_coordinator<T>>;
template <typename T>
using const_host_view = const_array_view<T, host_coordinator<T>>;

template <typename T>
std::ostream& operator<< (std::ostream& o, host_view<T> const& v) {
    o << "[";
    for(auto const& value: v) o << value << ",";
    o << "]";

    return o;
}

// specialization for pinned vectors. Use a host_coordinator, because memory is
// in the host memory space, and all of the helpers (copy, set, etc) are the
// same with and without page locked memory
template <typename T>
using pinned_vector = array<T, host_coordinator<T, pinned_allocator<T>>>;
template <typename T>
using pinned_view = array_view<T, host_coordinator<T, pinned_allocator<T>>>;

// specialization for device memory
template <typename T>
using device_vector = array<T, device_coordinator<T, gpu_allocator<T>>>;
template <typename T>
using device_view = array_view<T, device_coordinator<T, gpu_allocator<T>>>;
template <typename T>
using const_device_view = const_array_view<T, device_coordinator<T, gpu_allocator<T>>>;

template <typename T>
std::ostream& operator<<(std::ostream& o, device_view<T> v) {
    std::size_t i=0u;
    for (; i<v.size()-1; ++i) o << v[i] << ", ";
    return o << v[i];
}
template <typename T>
std::ostream& operator<<(std::ostream& o, const_device_view<T> v) {
    std::size_t i=0u;
    for (; i<v.size()-1; ++i) o << v[i] << ", ";
    return o << v[i];
}

} // namespace memory
} // namespace arb

// now import the helpers
// these require that host_vector etc have been defined
#include "wrappers.hpp"
#include "copy.hpp"
