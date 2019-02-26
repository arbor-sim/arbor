#pragma once

/* Represent a possibly-uninitialized value, reference or void.
 *
 * The uninitialized<X> structure holds space for an item of
 * type X, leaving its construction or destruction to the user.
 *
 * Specialisations for reference types X& and for the void type
 * allow for the handling of non-value types in a uniform manner.
 */

#include <type_traits>
#include <utility>

#include <arbor/util/compat.hpp>

namespace arb {
namespace util {

template <typename T>
using enable_if_copy_constructible_t =
    std::enable_if_t<std::is_copy_constructible<T>::value>;

template <typename... T>
using enable_if_constructible_t =
    std::enable_if_t<std::is_constructible<T...>::value>;

/*
 * Maintains storage for a value of type X, with explicit
 * construction and destruction.
 */
template <typename X>
class uninitialized {
private:
    std::aligned_storage_t<sizeof(X), alignof(X)> data;

public:
    using pointer = X*;
    using const_pointer = const X*;
    using reference = X&;
    using const_reference= const X&;
    using rvalue_reference = X&&;
    using const_rvalue_reference= const X&&;

    pointer ptr() {
        return reinterpret_cast<X*>(&data);
    }
    const_pointer cptr() const {
        return reinterpret_cast<const X*>(&data);
    }

    reference ref() { return *ptr(); }
    const_reference cref() const { return *cptr(); }

    // Copy construct the value.
    template <
        typename Y = X,
        typename = enable_if_copy_constructible_t<Y>
    >
    void construct(const X& x) {
        new(&data) X(x);
    }

    // General constructor for X, forwarding arguments.
    template <
        typename... Y,
        typename = enable_if_constructible_t<X, Y...>
    >
    void construct(Y&& ...args) {
        new(&data) X(std::forward<Y>(args)...);
    }

    void destruct() { ptr()->~X(); }

    // Apply the one-parameter functor F to the value by reference.
    template <typename F>
    std::result_of_t<F(reference)> apply(F&& f) { return f(ref()); }

    // Apply the one-parameter functor F to the value by const reference.
    template <typename F>
    std::result_of_t<F(const_reference)> apply(F&& f) const { return f(cref()); }
};

/*
 * Maintains storage for a pointer of type X, representing
 * a possibly uninitialized reference.
 */
template <typename X>
class uninitialized<X&> {
private:
    X *data;

public:
    using pointer = X*;
    using const_pointer = const X*;
    using reference = X&;
    using const_reference = const X&;
    using rvalue_reference = X&;
    using const_rvalue_reference= const X&;

    pointer ptr() { return data; }
    const_pointer cptr() const { return data; }

    reference ref() { return *data; }
    const_reference cref() const { return *data; }

    void construct(X& x) { data = &x; }
    void destruct() {}
};

/* Wrap a void type in an uninitialized template.
 *
 * Allows the use of uninitialized<X> for void X, for generic applications.
 */
template <>
class uninitialized<void> {
public:
    using pointer = void*;
    using const_pointer = const void*;
    using reference = void;
    using const_reference = void;
    using rvalue_reference = void;
    using const_rvalue_reference = void;

    pointer ptr() { return nullptr; }
    const_pointer cptr() const { return nullptr; }

    reference ref() {}
    const_reference cref() const {}

    // No operation.
    void construct(...) {}
    // No operation.
    void destruct() {}
};

} // namespace util
} // namespace arb
