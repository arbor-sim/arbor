#pragma once

/*
 * Utilities and base classes to help with
 * implementing iterators and iterator adaptors.
 */

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "util/meta.hpp"

namespace arb {
namespace util {

/*
 * Return the iterator reachable from iter such that
 * std::next(iter)==end
 *
 * Two implementations: the first applies generally, while the
 * second is used when we can just return std::prev(end).
 */
template <typename I, typename E>
std::enable_if_t<
    is_forward_iterator<I>::value &&
        (!is_bidirectional_iterator<E>::value || !std::is_constructible<I, E>::value),
    I>
upto(I iter, E end) {
    I j = iter;
    while (j!=end) {
        iter = j;
        ++j;
    }
    return iter;
}

template <typename I, typename E>
std::enable_if_t<is_bidirectional_iterator<E>::value && std::is_constructible<I, E>::value, I>
upto(I iter, E end) {
    return iter==I{end}? iter: I{std::prev(end)};
}

template <typename I, typename E,
          typename C = common_random_access_iterator_t<I,E>>
std::enable_if_t<std::is_same<I, E>::value ||
            (has_common_random_access_iterator<I,E>::value &&
             is_forward_iterator<I>::value),
            typename std::iterator_traits<C>::difference_type>
distance(I first, E last) {
    return std::distance(static_cast<C>(first), static_cast<C>(last));
}

template <typename I, typename E>
std::enable_if_t<!has_common_random_access_iterator<I, E>::value &&
            is_forward_iterator<I>::value,
            typename std::iterator_traits<I>::difference_type>
distance(I first, E last) {
    typename std::iterator_traits<I>::difference_type ret = 0;
    while (first != last) {
        ++first;
        ++ret;
    }

    return ret;
}

/*
 * generic front() and back() methods for containers or ranges
 */

template <typename Seq>
decltype(auto) front(Seq& seq) {
    using std::begin;
    return *begin(seq);
}

template <typename Seq>
decltype(auto) back(Seq& seq) {
    using std::begin;
    using std::end;

    return *upto(begin(seq), end(seq));
}

/*
 * Provide a proxy object for operator->() for iterator adaptors that
 * present rvalues on dereference.
 */
template <typename V>
struct pointer_proxy {
    V v;

    template <typename... Args>
    pointer_proxy(Args&&... args): v(std::forward<Args>(args)...) {}

    V* operator->() { return &v; }
    const V* operator->() const { return &v; }
};

/*
 * Base class (using CRTP) for iterator adaptors that
 * perform a transformation or present a proxy for
 * an underlying iterator.
 *
 * Supplies default implementations for iterator concepts
 * in terms of the derived class' methods and the
 * inner iterator.
 *
 * Derived class must provide implementations for:
 *   operator*()
 *   operator[](difference_type)
 *   inner()   // provides access to wrapped iterator
 */

template <typename Derived, typename I>
class iterator_adaptor {
protected:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

private:
    // Access to inner iterator provided by derived class.
    I& inner() { return derived().inner(); }
    const I& inner() const { return derived().inner(); }

public:
    using value_type = typename std::iterator_traits<I>::value_type;
    using difference_type = typename std::iterator_traits<I>::difference_type;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;
    using pointer = typename std::iterator_traits<I>::pointer;
    using reference = typename std::iterator_traits<I>::reference;

    iterator_adaptor() = default;

    // forward and input iterator requirements

    I operator->() { return inner(); }
    I operator->() const { return inner(); }

    Derived& operator++() {
        ++inner();
        return derived();
    }

    Derived operator++(int) {
        Derived c(derived());
        ++derived();
        return c;
    }

    bool operator==(const Derived& x) const {
        return inner()==x.inner();
    }

    bool operator!=(const Derived& x) const {
        return !(derived()==x);
    }

    // bidirectional iterator requirements

    Derived& operator--() {
        --inner();
        return derived();
    }

    Derived operator--(int) {
        Derived c(derived());
        --derived();
        return c;
    }

    // random access iterator requirements

    Derived& operator+=(difference_type n) {
        inner() += n;
        return derived();
    }

    Derived operator+(difference_type n) const {
        Derived c(derived());
        return c += n;
    }

    friend Derived operator+(difference_type n, const Derived& x) {
        return x+n;
    }

    Derived& operator-=(difference_type n) {
        inner() -= n;
        return *this;
    }

    Derived operator-(difference_type n) const {
        Derived c(derived());
        return c -= n;
    }

    difference_type operator-(const Derived& x) const {
        return inner()-x.inner();
    }

    bool operator<(const Derived& x) const {
        return inner()<x.inner();
    }

    bool operator<=(const Derived& x) const {
        return derived()<x || derived()==x;
    }

    bool operator>=(const Derived& x) const {
        return !(derived()<x);
    }

    bool operator>(const Derived& x) const {
        return !(derived()<=x);
    }
};

} // namespace util
} // namespace arb
