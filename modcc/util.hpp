#pragma once

#include <type_traits>
#include <initializer_list>

namespace impl {
    template <typename C, typename V>
    struct has_count_method {
        template <typename T, typename U>
        static decltype(std::declval<T>().count(std::declval<U>()), std::true_type{}) test(int);
        template <typename T, typename U>
        static std::false_type test(...);

        using type = decltype(test<C, V>(0));
    };

    template <typename X, typename C>
    bool is_in(const X& x, const C& c, std::false_type) {
        for (const auto& y: c) {
            if (y==x) return true;
        }
        return false;
    }

    template <typename X, typename C>
    bool is_in(const X& x, const C& c, std::true_type) {
        return !!c.count(x);
    }
}

template <typename X, typename C>
bool is_in(const X& x, const C& c) {
    return impl::is_in(x, c, typename impl::has_count_method<C,X>::type{});
}

template <typename X>
bool is_in(const X& x, const std::initializer_list<X>& c) {
    return impl::is_in(x, c, std::false_type{});
}

