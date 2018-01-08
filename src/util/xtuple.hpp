#pragma once

// Tuple-alike that allows implicit initialization from a brace-enclosed list.
// This functionality is offered by std::tuple in C++17 and is an extension
// in recent versions of libstdc++ and libc++.

#include <tuple>
#include <type_traits>

namespace arb {
namespace util {

namespace impl {
    template <typename... Ts>
    struct typelist;

    template <typename, typename>
    struct all_convertible: std::false_type {};

    template <>
    struct all_convertible<typelist<>, typelist<>>: std::true_type {};

    template <typename T, typename... Ts, typename U, typename... Us>
    struct all_convertible<typelist<T, Ts...>, typelist<U, Us...>>:
        std::conditional<std::is_convertible<T, U>::value, typename all_convertible<typelist<Ts...>, typelist<Us...>>::type, std::false_type>::type {};
}

template <typename... Ts>
struct xtuple: std::tuple<Ts...> {
    xtuple() = default;

    template <typename... Us, typename =
        typename std::enable_if<impl::all_convertible<impl::typelist<Us...>, impl::typelist<Ts...>>::value>::type>
    xtuple(Us&&... us): std::tuple<Ts...>(std::forward<Us>(us)...) {}
};

} // namespace util
} // namespace arb
