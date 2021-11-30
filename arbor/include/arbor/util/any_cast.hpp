#pragma once

// arb::util::any_cast wraps std::any_cast for std::any objects.
//
// arb::util::any_cast also has specializations for arb::util::unique_any
// and arb::util::any_pointer defined in the corresponding headers.

#include <any>
#include <type_traits>
#include <utility>

namespace arb {
namespace util {

template <
    typename T,
    typename Any,
    typename = std::enable_if_t<std::is_same_v<std::any, std::remove_cv_t<std::remove_reference_t<Any>>>>
>
T any_cast(Any&& a) { return std::any_cast<T>(std::forward<Any>(a)); }

template <typename T>
const T* any_cast(const std::any* p) noexcept { return std::any_cast<T>(p); }

template <typename T>
T* any_cast(std::any* p) noexcept { return std::any_cast<T>(p); }

} // namespace util
} // namespace arb
