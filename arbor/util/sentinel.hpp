#pragma once

#include <type_traits>
#include <variant>

#include "util/meta.hpp"

/*
 * Use a proxy iterator to present a range as having the same begin and
 * end types, for use with e.g. pre-C++17 ranged-for loops or STL
 * algorithms.
 */

namespace arb {
namespace util {

template<typename I, typename S, bool Same>
struct iterator_category_select {
    // default category
    using iterator_category = std::forward_iterator_tag;
};

template<typename I, typename S>
struct iterator_category_select<I,S,true> {
    using iterator_category = typename std::iterator_traits<I>::iterator_category;
};

template <typename I, typename S>
class sentinel_iterator {
    std::variant<I, S> e_;

    I& iter() {
        arb_assert(!is_sentinel());
        return std::get<0>(e_);
    }

    const I& iter() const {
        arb_assert(!is_sentinel());
        return std::get<0>(e_);
    }

    S& sentinel() {
        arb_assert(is_sentinel());
        return std::get<1>(e_);
    }

    const S& sentinel() const {
        arb_assert(is_sentinel());
        return std::get<1>(e_);
    }

public:
    using difference_type = typename std::iterator_traits<I>::difference_type;
    using value_type = typename std::iterator_traits<I>::value_type;
    using pointer = typename std::iterator_traits<I>::pointer;
    using reference = typename std::iterator_traits<I>::reference;
    using iterator_category = typename iterator_category_select<
        I,S,std::is_same<I,S>::value>::iterator_category;

    sentinel_iterator(I i): e_(i) {}

    template <typename V = S, typename = std::enable_if_t<!std::is_same<I, V>::value>>
    sentinel_iterator(S i): e_(i) {}

    sentinel_iterator() = default;
    sentinel_iterator(const sentinel_iterator&) = default;
    sentinel_iterator(sentinel_iterator&&) = default;

    sentinel_iterator& operator=(const sentinel_iterator&) = default;
    sentinel_iterator& operator=(sentinel_iterator&&) = default;

    // forward and input iterator requirements

    decltype(auto) operator*() const { return *iter(); }

    I operator->() const { return e_.template ptr<0>(); }

    sentinel_iterator& operator++() {
        ++iter();
        return *this;
    }

    sentinel_iterator operator++(int) {
        sentinel_iterator c(*this);
        ++*this;
        return c;
    }

    bool operator==(const sentinel_iterator& x) const {
        if (is_sentinel()) {
            return x.is_sentinel() || x.iter()==sentinel();
        }
        else {
            return x.is_sentinel()? iter()==x.sentinel(): iter()==x.iter();
        }
    }

    bool operator!=(const sentinel_iterator& x) const {
        return !(*this==x);
    }

    // bidirectional iterator requirements

    sentinel_iterator& operator--() {
        --iter();
        return *this;
    }

    sentinel_iterator operator--(int) {
        sentinel_iterator c(*this);
        --*this;
        return c;
    }

    // random access iterator requirements

    sentinel_iterator &operator+=(difference_type n) {
        iter() += n;
        return *this;
    }

    sentinel_iterator operator+(difference_type n) const {
        sentinel_iterator c(*this);
        return c += n;
    }

    friend sentinel_iterator operator+(difference_type n, sentinel_iterator x) {
        return x+n;
    }

    sentinel_iterator& operator-=(difference_type n) {
        iter() -= n;
        return *this;
    }

    sentinel_iterator operator-(difference_type n) const {
        sentinel_iterator c(*this);
        return c -= n;
    }

    difference_type operator-(sentinel_iterator x) const {
        return iter()-x.iter();
    }

    decltype(auto) operator[](difference_type n) const {
        return *(iter()+n);
    }

    bool operator<=(const sentinel_iterator& x) const {
        return x.is_sentinel() || (!is_sentinel() && iter()<=x.iter());
    }

    bool operator<(const sentinel_iterator& x) const {
        return !is_sentinel() && (x.is_sentinel() || iter()<=x.iter());
    }

    bool operator>=(const sentinel_iterator& x) const {
        return !(x<*this);
    }

    bool operator>(const sentinel_iterator& x) const {
        return !(x<=*this);
    }

    // access to underlying iterator/sentinel

    bool is_sentinel() const { return e_.index()!=0; }
    bool is_iterator() const { return e_.index()==0; }

    // default conversion to iterator, defined only if `is_iterator()` is true
    // (this really simplifies e.g. `maximum_element_by`, but it might not be
    // a super good idea.)
    operator I() const { return iter(); }
};

template <typename I, typename S>
using sentinel_iterator_t =
    std::conditional_t<std::is_same<I, S>::value, I, sentinel_iterator<I, S>>;

template <typename I, typename S>
sentinel_iterator_t<I, S> make_sentinel_iterator(const I& i, const S& s) {
    return sentinel_iterator_t<I, S>(i);
}

template <typename I, typename S>
sentinel_iterator_t<I, S> make_sentinel_end(const I& i, const S& s) {
    return sentinel_iterator_t<I, S>(s);
}


} // namespace util
} // namespace arb
