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
    if constexpr (std::is_convertible_v<T, std::string_view>) {
        constexpr std::size_t prime = 0x100000001b3;
        constexpr std::size_t offset_basis=0xcbf29ce484222325;

        std::size_t hash = offset_basis;

        for (uint8_t byte: std::string_view{data}) {
            hash = hash ^ byte;
            hash = hash * prime;
        }

        return hash;
    } else {
        return std::hash<std::decay_t<T>>{}(data);
    }
}

inline
std::size_t hash_value_combine(std::size_t n) { return n; }

template <typename... T>
std::size_t hash_value_combine(std::size_t n, std::size_t m, T... tail) {
    constexpr std::size_t prime2 = 54517;
    return hash_value_combine(prime2*n + m, tail...);
}

template <typename... T>
std::size_t hash_value(const T&... t) {
    constexpr std::size_t prime1 = 93481;
    return hash_value_combine(prime1, internal_hash(t)...);
}
}

#define ARB_DEFINE_HASH(type,...)\
namespace std {\
template <> struct hash<type> { std::size_t operator()(const type& a) const { return ::arb::hash_value(__VA_ARGS__); }};\
}
