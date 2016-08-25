#pragma once

/*
 * Present a single object as a (non-owning) container with one
 * element.
 *
 * (Will be subsumed by range/view code.)
 */

#include <algorithm>

namespace nest {
namespace mc {
namespace util {

template <typename X>
struct singleton_adaptor {
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = X;
    using reference = X&;
    using const_reference = const X&;
    using iterator = X*;
    using const_iterator = const X*;

    X* xp;
    singleton_adaptor(X& x): xp(&x) {}

    const X* cbegin() const { return xp; }
    const X* begin() const { return xp; }
    X* begin() { return xp; }

    const X* cend() const { return xp+1; }
    const X* end() const { return xp+1; }
    X* end() { return xp+1; }

    std::size_t size() const { return 1u; }

    bool empty() const { return false; }

    const X* front() const { return *xp; }
    X* front() { return *xp; }

    const X* back() const { return *xp; }
    X* back() { return *xp; }

    const X* operator[](difference_type) const { return *xp; }
    X* operator[](difference_type) { return *xp; }

    void swap(singleton_adaptor& s) { std::swap(xp, s.xp); }
    friend void swap(singleton_adaptor& r, singleton_adaptor& s) {
        r.swap(s);
    }
};

template <typename X>
singleton_adaptor<X> singleton_view(X& x) {
    return singleton_adaptor<X>(x);
}

} // namespace util
} // namespace mc
} // namespace nest
