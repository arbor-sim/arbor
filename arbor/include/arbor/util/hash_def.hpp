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
#include <typeindex>

// Helpers for forming hash values of compounds objects.

namespace arb {

inline std::size_t hash_value_combine(std::size_t n) {
    return n;
}

template <typename... T>
std::size_t hash_value_combine(std::size_t n, std::size_t m, T... tail) {
    constexpr std::size_t prime2 = 54517;
    return hash_value_combine(prime2*n + m, tail...);
}

template <typename... T>
std::size_t hash_value(const T&... t) {
    constexpr std::size_t prime1 = 93481;
    return hash_value_combine(prime1, std::hash<T>{}(t)...);
}
}

#define ARB_DEFINE_HASH(type,...)\
namespace std {\
template <> struct hash<type> { std::size_t operator()(const type& a) const { return ::arb::hash_value(__VA_ARGS__); }};\
}
