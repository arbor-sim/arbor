#pragma once

/*
 * Macro definitions for defining hash functions for compound objects.
 *
 * Use:
 *
 * To define a std::hash overload for a record type xyzzy
 * with fields foo, bar and baz:
 *
 * ARB_DEFINE_HASH(xyzzy, a.foo, a.bar, a.baz)
 *
 * The explicit use of 'a' in the macro invocation is just to simplify
 * the implementation.
 *
 * The macro must be used outside of any namespace.
 */

#include <cstddef>
#include <string_view>
#include <functional>

#include <iostream>

// Helpers for forming hash values of compounds objects.

namespace arb {

// Non-cryptographic hash function for mapping strings to internal
// identifiers. Concretely, FNV-1a hash function taken from
//
//   http://www.isthe.com/chongo/tech/comp/fnv/index.html
//
// NOTE: It may be worth it considering different hash functions in
//       the future that have better characteristic, xxHash or Murmur
//       look interesting but are more complex and likely require adding
//       external dependencies.
//       NOTE: this is the obligatory comment on a better hash function
//             that will be here until the end of time.

template <typename T>
inline constexpr std::size_t internal_hash(T&& data) {
    using D = std::decay_t<T>;
    constexpr std::size_t prime = 0x100000001b3;
    constexpr std::size_t offset_basis = 0xcbf29ce484222325;
    static_assert(!std::is_pointer_v<D> || std::is_same_v<D, void*> || std::is_convertible_v<T, std::string_view>,
                  "Pointer types except void* will not be hashed.");
    if constexpr (std::is_convertible_v<T, std::string_view>) {
        std::size_t hash = offset_basis;
        for (uint8_t byte: std::string_view{data}) {
            hash = hash ^ byte;
            hash = hash * prime;
        }
        return hash;
    }
    if constexpr (std::is_integral_v<D>) {
        unsigned long long bytes = data;
        std::size_t hash = offset_basis;
        for (int ix = 0; ix < sizeof(data); ++ix) {
            uint8_t byte = bytes & 255;
            bytes >>= 8;
            hash = hash ^ byte;
            hash = hash * prime;
        }
        return hash;
    }
    if constexpr (std::is_pointer_v<D>) {
        unsigned long long bytes = reinterpret_cast<unsigned long long>(data);
        std::size_t hash = offset_basis;
        for (int ix = 0; ix < sizeof(data); ++ix) {
            uint8_t byte = bytes & 255;
            bytes >>= 8;
            hash = hash ^ byte;
            hash = hash * prime;
        }
        return hash;
    }
    return std::hash<D>{}(data);
}

inline
std::size_t hash_value_combine(std::size_t n) { return n; }

template <typename T, typename... Ts>
std::size_t hash_value_combine(std::size_t n, const T& head, const Ts&... tail) {
    constexpr std::size_t prime = 54517;
    return hash_value_combine(prime*n + internal_hash(head), tail...);
}

template <typename... T>
std::size_t hash_value(const T&... ts) {
    return hash_value_combine(0, ts...);
}
}

#define ARB_DEFINE_HASH(type,...)\
namespace std {\
template <> struct hash<type> { std::size_t operator()(const type& a) const { return ::arb::hash_value(__VA_ARGS__); }};\
}
