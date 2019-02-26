#pragma once

#include <type_traits>
#include <vector>

#include <memory/memory.hpp>

namespace arb {
namespace memory {

//
// helpers that allow us to easily generate a view or const_view for an arbitrary type
//

//
// array
//

// note we have to pass by non-const reference because
template <typename T, typename Coordinator>
array_view<T, Coordinator>
make_view(array<T, Coordinator>& a) {
    return array_view<T, Coordinator>(a.data(), a.size());
}

// pass by const reference
template <typename T, typename Coordinator>
const_array_view<T, Coordinator>
make_const_view(const array<T, Coordinator>& a) {
    return const_array_view<T, Coordinator>(a.data(), a.size());
}

//
// array_view
//
template <typename T, typename Coordinator>
array_view<T, Coordinator>
make_view(array_view<T, Coordinator> v) {
    return v;
}

template <typename T, typename Coordinator>
const_array_view<T, Coordinator>
make_const_view(array_view<T, Coordinator> v) {
    return const_array_view<T, Coordinator>(v.data(), v.size());
}

template <typename T, typename Coordinator>
const_array_view<T, Coordinator>
make_const_view(const_array_view<T, Coordinator> v) {
    return const_array_view<T, Coordinator>(v.data(), v.size());
}

//
// std::vector
//
template <typename T, typename Allocator>
array_view<T, host_coordinator<T>>
make_view(std::vector<T, Allocator>& vec) {
    return array_view<T, host_coordinator<T>>(vec.data(), vec.size());
}

template <typename T, typename Allocator>
const_array_view<T, host_coordinator<T>>
make_const_view(const std::vector<T, Allocator>& vec) {
    return const_array_view<T, host_coordinator<T>>(vec.data(), vec.size());
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
    struct is_on_host<array<T, host_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_host<array_view<T, host_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_host<const_array_view<T, host_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T>
    constexpr bool is_on_host_v() {
        return is_on_host<std::decay_t<T>>::value;
    }

    template <typename T>
    struct is_on_gpu : std::false_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<array<T, device_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<array_view<T, device_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T, typename Allocator>
    struct is_on_gpu<const_array_view<T, device_coordinator<T, Allocator>>> : std::true_type {};

    template <typename T>
    constexpr bool is_on_gpu_v() {
        return is_on_gpu<std::decay_t<T>>::value;
    }
}


//
// Helpers for getting a target-specific view of data.
// these return either const_view or an rvalue, so that the original memory
// range can't be modified via the returned type.
//

// host
template <
    typename C,
    typename = std::enable_if_t<util::is_on_host_v<C>()>
>
auto on_host(const C& c) {
    return make_const_view(c);
}

template <
    typename C,
    typename = std::enable_if_t<util::is_on_gpu_v<C>()>
>
auto on_host(const C& c) -> host_vector<typename C::value_type> {
    using T = typename C::value_type;
    return host_vector<T>(make_const_view(c));
}

// gpu
template <
    typename C,
    typename = std::enable_if_t<util::is_on_gpu_v<C>()>
>
auto on_gpu(const C& c) -> decltype(make_const_view(c)) {
    return make_const_view(c);
}

template <
    typename C,
    typename = std::enable_if_t<util::is_on_host_v<C>()>
>
auto on_gpu(const C& c) -> device_vector<typename C::value_type> {
    using T = typename C::value_type;
    return device_vector<T>(make_const_view(c));
}

} // namespace memory
} // namespace arb
