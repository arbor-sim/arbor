#pragma once

#include <memory>

namespace arb {
namespace util {

// just because we aren't using C++14, doesn't mean we shouldn't go
// without make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

} // namespace util
} // namespace arb


