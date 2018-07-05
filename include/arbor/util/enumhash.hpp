#pragma once

#include <functional>
#include <type_traits>

// Work around for C++11 defect #2148: hashing enums should be supported directly by std::hash.
// Fixed in C++14.

namespace arb {
namespace util {

struct enum_hash {
    template <typename E, typename V = std::underlying_type_t<E>>
    std::size_t operator()(E e) const noexcept {
        return std::hash<V>{}(static_cast<V>(e));
    }
};

} // namespace util
} // namespace arb
