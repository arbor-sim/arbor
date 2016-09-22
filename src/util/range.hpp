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
#include <util/sentinel.hpp>

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

    bool empty() const { return left == right; }

    iterator begin() const { return left; }
    const_iterator cbegin() const { return left; }

    sentinel end() const { return right; }
    sentinel cend() const { return right; }

    template <typename V = iterator>
    enable_if_t<is_forward_iterator<V>::value, size_type>
    size() const {
        return util::distance(begin(), end());
    }

    constexpr size_type max_size() const {
        return std::numeric_limits<size_type>::max();
    }

    void swap(range& other) {
        std::swap(left, other.left);
        std::swap(right, other.right);
    }

    auto front() const -> decltype(*left) { return *left; }

    auto back() const -> decltype(*left) { return *upto(left, right); }

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

    static constexpr bool is_splittable_in_proportion() {
        return is_forward_iterator<U>::value;
    }
#endif
};

template <typename U, typename V>
range<U, V> make_range(const U& left, const V& right) {
    return range<U, V>(left, right);
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
Container& append(Container &c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.insert(c.end(), std::begin(canon), std::end(canon));
    return c;
}

template <typename AssignableContainer, typename Seq>
AssignableContainer& assign(AssignableContainer& c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.assign(std::begin(canon), std::end(canon));
    return c;
}

template <typename Seq>
range<typename sequence_traits<Seq>::iterator_type, typename sequence_traits<Seq>::sentinel_type>
range_view(Seq& seq) {
    return make_range(std::begin(seq), std::end(seq));
}

template <
    typename Seq,
    typename Iter = typename sequence_traits<Seq>::iterator_type,
    typename Size = typename sequence_traits<Seq>::size_type
>
enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq& seq, Size bi, Size ei) {
    Iter b = std::next(std::begin(seq), bi);
    Iter e = std::next(b, ei-bi);
    return make_range(b, e);
}

} // namespace util
} // namespace mc
} // namespace nest
