#pragma once

// Provides `overload` function for constructing a functor with overloaded
// `operator()` from a number of functions or function objects.
//
// Provides the `any_visitor` class, which will call a provided functional
// with the contained value in a `std::any` if it is one of a fixed set
// of types.

#include <any>
#include <utility>
#include <type_traits>

namespace arb {
namespace util {

namespace impl {

template <typename X, typename Y>
struct propagate_qualifier { using type = Y; };

template <typename X, typename Y>
struct propagate_qualifier<const X, Y> { using type = const Y; };

template <typename X, typename Y>
struct propagate_qualifier<X&, Y> { using type = Y&; };

template <typename X, typename Y>
struct propagate_qualifier<const X&, Y> { using type = const Y&; };

template <typename X, typename Y>
using propagate_qualifier_t = typename propagate_qualifier<X, Y>::type;

} // namespace impl

// A type `any_visitor<A, B, ...>` has one public static method
// `visit(f, a)` where `a` is a possibly const lvalue or rvalue std::any,
// and `f` is a functional object or function pointer.
//
// If `a` contains a value of any of the types `A, B, ...`, `f` will
// be called with that value. If `a` is an lvalue, it will be passed by
// lvaue reference; otherwise it will be moved.
//
// If `a` contains no value, or a value not in the type list `A, B, ...`,
// then it will evaluate `f()` if it is defined, or else throw a
// `bad_any_cast` exception.

template <typename, typename...> struct any_visitor;

template <typename T>
struct any_visitor<T> {
    template <typename F, typename = void>
    struct invoke_or_throw {
        template <typename A>
        static auto visit(F&& f, A&& a) {
            using Q = impl::propagate_qualifier_t<A, T>;
            return std::any_cast<T>(&a)?
                   std::forward<F>(f)(std::any_cast<Q&&>(std::forward<A>(a))):
                   throw std::bad_any_cast();
        }
    };

    template <typename F>
    struct invoke_or_throw<F, std::void_t<decltype(std::declval<F>()())>> {
        template <typename A>
        static auto visit(F&& f, A&& a) {
            using Q = impl::propagate_qualifier_t<A, T>;
            return std::any_cast<T>(&a)?
                   std::forward<F>(f)(std::any_cast<Q&&>(std::forward<A>(a))):
                   std::forward<F>(f)();
        }
    };

    template <typename F, typename A,
        typename = std::enable_if_t<std::is_same_v<std::any, std::decay_t<A>>>
    >
    static auto visit(F&& f, A&& a) {
        return invoke_or_throw<F>::visit(std::forward<F>(f), std::forward<A>(a));
    }
};

template <typename T, typename U, typename... Rest>
struct any_visitor<T, U, Rest...> {
    template <typename F, typename A,
        typename = std::enable_if_t<std::is_same_v<std::any, std::decay_t<A>>>
    >
    static auto visit(F&& f, A&& a) {
        using Q = impl::propagate_qualifier_t<A, T>;
        return std::any_cast<T>(&a)?
               std::forward<F>(f)(std::any_cast<Q&&>(std::forward<A>(a))):
               any_visitor<U, Rest...>::visit(std::forward<F>(f), std::forward<A>(a));
    }
};

namespace impl {

template <typename, typename...> struct overload_impl {};

template <typename F1>
struct overload_impl<F1> {
    F1 f_;

    overload_impl(F1&& f1): f_(std::forward<F1>(f1)) {}

    template <typename... A, std::enable_if_t<std::is_invocable_v<F1, A...>, int> = 0>
    decltype(auto) operator()(A&&... a) { return f_(std::forward<A>(a)...); }
};

template <typename F1, typename F2, typename... Fn>
struct overload_impl<F1, F2, Fn...>: overload_impl<F2, Fn...> {
    F1 f_;

    overload_impl(F1&& f1, F2&& f2, Fn&&... fn):
        overload_impl<F2, Fn...>(std::forward<F2>(f2), std::forward<Fn>(fn)...),
        f_(std::forward<F1>(f1)) {}

    template <typename... A, std::enable_if_t<std::is_invocable_v<F1, A...>, int> = 0>
    decltype(auto) operator()(A&&... a) { return f_(std::forward<A>(a)...); }

    template <typename... A, std::enable_if_t<!std::is_invocable_v<F1, A...>, int> = 0>
    decltype(auto) operator()(A&&... a) {
        return overload_impl<F2, Fn...>::operator()(std::forward<A>(a)...);
    }
};

} // namespace impl


// `overload(f, g, h, ...)` returns a functional object whose `operator()` is overloaded
// with those of `f`, `g`, `h`, ... in decreasing order of precedence.

template <typename... Fn>
auto overload(Fn&&... fn) {
    return impl::overload_impl<Fn...>(std::forward<Fn>(fn)...);
}

} // namespace util
} // namespace arb
