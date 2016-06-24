#pragma once

/* Type utilities and convenience expressions.  */

#include <type_traits>

namespace nest {
namespace mc {
namespace util {

// Until C++14 ...

template <typename T>
using result_of_t = typename std::result_of<T>::type;

template <bool V>
using enable_if_t = typename std::enable_if<V>::type;

// Convenience short cuts

template <typename T>
using enable_if_copy_constructible_t =
    enable_if_t<std::is_copy_constructible<T>::value>;

template <typename T>
using enable_if_move_constructible_t =
    enable_if_t<std::is_move_constructible<T>::value>;

template <typename... T>
using enable_if_constructible_t =
    enable_if_t<std::is_constructible<T...>::value>;


} // namespace util
} // namespace mc
} // namespace nest
