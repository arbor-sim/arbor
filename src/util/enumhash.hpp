#pragma once

// Work around for C++11 defect #2148: hashing enums should be supported directly by std::hash.
// Fixed in C++14.

namespace arb {
namespace util {

struct enum_hash {
    template <typename E, typename V = typename std::underlying_type<E>::type>
    std::size_t operator()(E e) const noexcept {
        return std::hash<V>{}(static_cast<V>(e));
    }
};

} // namespace util
} // namespace arb
