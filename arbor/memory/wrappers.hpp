#pragma once

#include <type_traits>
#include <vector>

#include <memory/memory.hpp>

#include <arbor/serdes.hpp>

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

template<typename K,
         typename T>
void serialize(::arb::serializer ser, const K& k, const memory::host_vector<T>& hvs) {
    ser.begin_write_array(to_serdes_key(k));
    for (size_t ix = 0; ix < hvs.size(); ++ix) {
        serialize(ser, ix, hvs[ix]);
    }

    ser.end_write_array();
}

template<typename K,
         typename T>
void serialize(::arb::serializer ser, const K& k, const memory::device_vector<T>& vs) {
    auto hvs = on_host(vs);
    serialize(ser, k, hvs);
}

template<typename K,
         typename V>
void deserialize(::arb::serializer ser, const K& k, memory::host_vector<V>& hvs) {
    ser.begin_read_array(to_serdes_key(k));
    size_t ix = 0;
    for (; ser.next_key(); ++ix) {
        if (ix >= hvs.size()) throw std::runtime_error("Size mismatch");
        deserialize(ser, std::to_string(ix), hvs[ix]);
    }
    if (ix + 1 < hvs.size()) throw std::runtime_error("Size mismatch");
    ser.end_read_array();
}

template<typename K,
         typename V>
void deserialize(::arb::serializer ser, const K& k, memory::device_vector<V>& vs) {
    auto hvs = memory::host_vector<V>(vs.size());
    deserialize(ser, k, hvs);
    typename memory::host_vector<V>::coordinator_type{}.copy(hvs, vs);
}

} // namespace arb
