#pragma once

/* Present a pair of iterators as a non-owning collection.
 *
 * Two public member fields, `left` and `right`, describe
 * the half-open interval [`left`, `right`).
 *
 * Constness of the range object only affects mutability
 * of the iterators, and does not relate to the constness
 * of the data to which the iterators refer.
 *
 * The `right` field may differ in type from the `left` field,
 * in which case it is regarded as a sentinel type; the end of
 * the interval is then marked by the first successor `i` of
 * `left` that satisfies `i==right`.
 *
 * For an iterator `i` and sentinel `s`, it is expected that
 * the tests `i==s` and `i!=s` are well defined, with the
 * corresponding semantics.
 */

#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef WITH_TBB
#include <tbb/tbb_stddef.h>
#endif

#include <util/counter.hpp>
#include <util/debug.hpp>
#include <util/either.hpp>
#include <util/iterutil.hpp>
#include <util/meta.hpp>

namespace nest {
namespace mc {
namespace util {

template <typename U, typename S = U>
struct range {
    using iterator = U;
    using sentinel = S;
    using const_iterator = iterator;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = typename std::make_unsigned<difference_type>::type;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using reference = typename std::iterator_traits<iterator>::reference;
    using const_reference = const value_type&;

    iterator left;
    sentinel right;

    range() = default;
    range(const range&) = default;
    range(range&&) = default;

    template <typename U1, typename U2>
    range(U1&& l, U2&& r):
        left(std::forward<U1>(l)), right(std::forward<U2>(r))
    {}

    range& operator=(const range&) = default;
    range& operator=(range&&) = default;

    bool empty() const { return left==right; }

    iterator begin() const { return left; }
    const_iterator cbegin() const { return left; }

    sentinel end() const { return right; }
    sentinel cend() const { return right; }

    template <typename V = iterator>
    enable_if_t<is_forward_iterator<V>::value, size_type>
    size() const {
        return std::distance(begin(), end());
    }

    constexpr size_type max_size() const { return std::numeric_limits<size_type>::max(); }

    void swap(range& other) {
        std::swap(left, other.left);
        std::swap(right, other.right);
    }

    decltype(*left) front() const { return *left; }

    decltype(*left) back() const { return *upto(left, right); }

    template <typename V = iterator>
    enable_if_t<is_random_access_iterator<V>::value, decltype(*left)>
    operator[](difference_type n) const {
        return *std::next(begin(), n);
    }

    template <typename V = iterator>
    enable_if_t<is_random_access_iterator<V>::value, decltype(*left)>
    at(difference_type n) const {
        if (size_type(n) >= size()) {
            throw std::out_of_range("out of range in range");
        }
        return (*this)[n];
    }

#ifdef WITH_TBB
    template <
        typename V = iterator,
        typename = enable_if_t<is_forward_iterator<V>::value>
    >
    range(range& r, tbb::split):
        left(r.left), right(r.right)
    {
        std::advance(left, r.size()/2u);
        r.right = left;
    }

    template <
        typename V = iterator,
        typename = enable_if_t<is_forward_iterator<V>::value>
    >
    range(range& r, tbb::proportional_split p):
        left(r.left), right(r.right)
    {
        size_type i = (r.size()*p.left())/(p.left()+p.right());
        if (i<1) {
            i = 1;
        }
        std::advance(left, i);
        r.right = left;
    }

    bool is_divisible() const {
        return is_forward_iterator<U>::value && left != right && std::next(left) != right;
    }

    static const bool is_splittable_in_proportion() {
        return is_forward_iterator<U>::value;
    }
#endif
};

template <typename U, typename V>
range<U, V> make_range(const U& left, const V& right) {
    return range<U, V>(left, right);
}

/*
 * Use a proxy iterator to present a range as having the same begin and
 * end types, for use with e.g. pre-C++17 ranged-for loops or STL
 * algorithms.
 */
template <typename I, typename S>
class sentinel_iterator {
    nest::mc::util::either<I, S> e_;

    bool is_sentinel() const { return e_.index()!=0; }

    I& iter() {
        EXPECTS(!is_sentinel());
        return e_.template unsafe_get<0>();
    }

    const I& iter() const {
        EXPECTS(!is_sentinel());
        return e_.template unsafe_get<0>();
    }

    S& sentinel() {
        EXPECTS(is_sentinel());
        return e_.template unsafe_get<1>();
    }

    const S& sentinel() const {
        EXPECTS(is_sentinel());
        return e_.template unsafe_get<1>();
    }

public:
    using difference_type = typename std::iterator_traits<I>::difference_type;
    using value_type = typename std::iterator_traits<I>::value_type;
    using pointer = typename std::iterator_traits<I>::pointer;
    using reference = typename std::iterator_traits<I>::reference;
    using iterator_category = typename std::iterator_traits<I>::iterator_category;

    sentinel_iterator(I i): e_(i) {}

    template <typename V = S, typename = enable_if_t<!std::is_same<I, V>::value>>
    sentinel_iterator(S i): e_(i) {}

    sentinel_iterator() = default;
    sentinel_iterator(const sentinel_iterator&) = default;
    sentinel_iterator(sentinel_iterator&&) = default;

    sentinel_iterator& operator=(const sentinel_iterator&) = default;
    sentinel_iterator& operator=(sentinel_iterator&&) = default;

    // forward and input iterator requirements

    auto operator*() const -> decltype(*iter()) { return *iter(); }

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

    auto operator[](difference_type n) const -> decltype(*iter()){
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
};

template <typename I, typename S>
using sentinel_iterator_t =
    typename std::conditional<std::is_same<I, S>::value, I, sentinel_iterator<I, S>>::type;

template <typename I, typename S>
sentinel_iterator_t<I, S> make_sentinel_iterator(const I& i, const S& s) {
    return sentinel_iterator_t<I, S>(i);
}

template <typename I, typename S>
sentinel_iterator_t<I, S> make_sentinel_end(const I& i, const S& s) {
    return sentinel_iterator_t<I, S>(s);
}

template <typename Seq>
auto canonical_view(const Seq& s) ->
    range<sentinel_iterator_t<decltype(std::begin(s)), decltype(std::end(s))>>
{
    return {make_sentinel_iterator(std::begin(s), std::end(s)), make_sentinel_end(std::begin(s), std::end(s))};
}

/*
 * Present a single item as a range
 */

template <typename T>
range<T*> singleton_view(T& item) {
    return {&item, &item+1};
}

template <typename T>
range<const T*> singleton_view(const T& item) {
    return {&item, &item+1};
}

/*
 * Range/container utility functions
 */

template <typename Container, typename Seq>
void append(Container &c, const Seq& seq) {
    c.insert(c.end(), seq.begin(), seq.end());
}

} // namespace util
} // namespace mc
} // namespace nest
