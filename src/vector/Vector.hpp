#pragma once

#include <iostream>

#include "Array.hpp"
#include "definitions.hpp"
#include "HostCoordinator.hpp"

#ifdef WITH_CUDA
#include "DeviceCoordinator.hpp"
#endif


namespace memory {

// specialization for host vectors
template <typename T>
using HostVector = Array<T, HostCoordinator<T>>;
template <typename T>
using HostView = ArrayView<T, HostCoordinator<T>>;

template <typename T>
std::ostream& operator<< (std::ostream& o, HostView<T> const& v) {
    o << "[";
    for(auto const& value: v) o << value << ",";
    o << "]";

    return o;
}

#ifdef WITH_CUDA
// specialization for pinned vectors. Use a host_coordinator, because memory is
// in the host memory space, and all of the helpers (copy, set, etc) are the
// same with and without page locked memory
template <typename T>
using PinnedVector = Array<T, HostCoordinator<T, PinnedAllocator<T>>>;
template <typename T>
using PinnedView = ArrayView<T, HostCoordinator<T, PinnedAllocator<T>>>;

// specialization for device memory
template <typename T>
using DeviceVector = Array<T, DeviceCoordinator<T, CudaAllocator<T>>>;
template <typename T>
using DeviceView = ArrayView<T, DeviceCoordinator<T, CudaAllocator<T>>>;
#endif

#ifdef WITH_KNL
// specialization for HBW memory on KNL
template <typename T>
using HBWVector = Array<T, HostCoordinator<T, HBWAllocator<T>>>;
template <typename T>
using HBWView = ArrayView<T, HostCoordinator<T, HBWAllocator<T>>>;
#endif

} // namespace memory
