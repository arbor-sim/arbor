#pragma once

#include <memory>

namespace arb {
namespace util {

// TODO: Remove when migrate to C++14

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

} // namespace util
} // namespace arb


