#pragma once

#include <vector>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace memory {

//
// helpers that allow us to easily generate a view or const_view for
// an arbitrary type
//

//
// Array
//

// pass by non-const reference because it is not possible to make a view from a const Array
template <typename T, typename Coordinator>
ArrayView<T, Coordinator>
make_view(Array<T, Coordinator>& a) {
    return ArrayView<T, Coordinator>(a.data(), a.size());
}

// pass by const reference because both const and non-const can be cast to const
template <typename T, typename Coordinator>
ConstArrayView<T, Coordinator>
make_const_view(const Array<T, Coordinator>& a) {
    return ConstArrayView<T, Coordinator>(a.data(), a.size());
}

//
// ArrayView
//
template <typename T, typename Coordinator>
ArrayView<T, Coordinator>
make_view(ArrayView<T, Coordinator> v) {
    return v;
}

template <typename T, typename Coordinator>
ConstArrayView<T, Coordinator>
make_const_view(ArrayView<T, Coordinator> v) {
    return v;
}

//
// std::vector
//
template <typename T, typename Allocator>
ArrayView<T, HostCoordinator<T>>
make_view(std::vector<T, Allocator>& vec) {
    return ArrayView<T, HostCoordinator<T>>(vec.data(), vec.size());
}

template <typename T, typename Allocator>
ConstArrayView<T, HostCoordinator<T>>
make_const_view(const std::vector<T, Allocator>& vec) {
    return ConstArrayView<T, HostCoordinator<T>>(vec.data(), vec.size());
}

//
// namespace with metafunctions
//
namespace util {
    template <typename T>
    struct is_on_host : std::false_type {};

    template <typename T, typename Allocator>
    struct is_on_host<std::vector<T, Allocator>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_host<Array<T, HostCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_host<ArrayView<T, HostCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_host<ConstArrayView<T, HostCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T>
    constexpr bool is_on_host_v() {
        return is_on_host<typename std::decay<T>::type>::value;
    }

    #ifdef WITH_CUDA
    template <typename T>
    struct is_on_gpu : std::false_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<Array<T, DeviceCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<ArrayView<T, DeviceCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<ConstArrayView<T, DeviceCoordinator<T, Allocator>>> : std::true_type {};

    template <typename T>
    constexpr bool is_on_gpu_v() {
        return is_on_gpu<typename std::decay<T>::type>::value;
    }
    #endif
}


//
// Helpers for getting a target-specific view of data.
// these return either const_view or an rvalue (i.e. the original memory range
// can't be modified via a type generated in this way)
//

// host
template <
    typename C,
    typename = typename std::enable_if<util::is_on_host_v<C>()>::type
>
auto on_host(const C& c) -> decltype(make_const_view(c)) {
    return make_const_view(c);
}

template <
    typename C,
    typename = typename std::enable_if<util::is_on_gpu_v<C>()>::type
>
auto on_host(const C& c) -> HostVector<typename C::value_type> {
    using T = typename C::value_type;
    return HostVector<T>(make_const_view(c));
}

// gpu
template <
    typename C,
    typename = typename std::enable_if<util::is_on_gpu_v<C>()>::type
>
auto on_gpu(const C& c) -> decltype(make_const_view(c)) {
    return make_const_view(c);
}

template <
    typename C,
    typename = typename std::enable_if<util::is_on_host_v<C>()>::type
>
auto on_gpu(const C& c) -> DeviceVector<typename C::value_type> {
    using T = typename C::value_type;
    return DeviceVector<T>(make_const_view(c));
}

/*
#ifdef WITH_CUDA
namespace util {
    template <typename T>
    bool is_host_pointer(const T* ptr) {
        cudaPointerAttributes attributes;
        // cast away constness for external C API call
        cudaPointerGetAttributes(
            &attributes,
            const_cast<void*>(static_cast<const void*>(ptr)));
        return attributes.memoryType == cudaMemoryTypeHost;
    }

    template <typename T>
    bool is_device_pointer(const T* ptr) {
        return !is_host_pointer(ptr);
    }
}
#endif
*/

} // namespace memory
