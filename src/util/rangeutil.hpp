#pragma once

/*
 * Sequence and container utilities compatible
 * with ranges.
 */

#include <algorithm>
#include <iterator>
#include <ostream>
#include <numeric>

#include <util/deduce_return.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>
#include <util/transform.hpp>
#include <util/meta.hpp>

namespace arb {
namespace util {

// Present a single item as a range

template <typename T>
range<T*> singleton_view(T& item) {
    return {&item, &item+1};
}

template <typename T>
range<const T*> singleton_view(const T& item) {
    return {&item, &item+1};
}

// Non-owning views and subviews

template <typename Seq>
range<typename sequence_traits<Seq>::iterator, typename sequence_traits<Seq>::sentinel>
range_view(Seq& seq) {
    return make_range(std::begin(seq), std::end(seq));
}

template <
    typename Seq,
    typename Offset1,
    typename Offset2,
    typename Iter = typename sequence_traits<Seq>::iterator
>
enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq& seq, Offset1 bi, Offset2 ei) {
    Iter b = std::begin(seq);
    std::advance(b, bi);

    Iter e = b;
    std::advance(e, ei-bi);
    return make_range(b, e);
}

template <
    typename Seq,
    typename Offset1,
    typename Offset2,
    typename Iter = typename sequence_traits<Seq>::iterator
>
enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq& seq, std::pair<Offset1, Offset2> index) {
    return subrange_view(seq, index.first, index.second);
}

// helper for determining the type of a subrange_view
template <typename Seq>
using subrange_view_type = decltype(subrange_view(std::declval<Seq&>(), 0, 0));


// Fill container or range.

template <typename Seq, typename V>
void fill(Seq& seq, const V& value) {
    auto canon = canonical_view(seq);
    std::fill(canon.begin(), canon.end(), value);
}

template <typename Range, typename V>
void fill(const Range& seq, const V& value) {
    auto canon = canonical_view(seq);
    std::fill(canon.begin(), canon.end(), value);
}

// Append sequence to a container

template <typename Container, typename Seq>
Container& append(Container &c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.insert(c.end(), std::begin(canon), std::end(canon));
    return c;
}

// Assign sequence to a container

template <typename AssignableContainer, typename Seq>
AssignableContainer& assign(AssignableContainer& c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.assign(std::begin(canon), std::end(canon));
    return c;
}

namespace impl {
    template <typename Seq>
    struct assign_proxy {
        assign_proxy(const Seq& seq):
            ref{seq}
        {}

        // Convert the sequence to a container of type C.
        // This requires that C supports construction from a pair of iterators
        template <typename C>
        operator C() const {
            return C(std::begin(ref), std::end(ref));
        }

        const Seq& ref;
    };
}

// Copy-assign sequence to a container

template <typename Seq>
impl::assign_proxy<Seq> assign_from(const Seq& seq) {
    return impl::assign_proxy<Seq>(seq);
}

// Assign sequence to a container with transform `proj`

template <typename AssignableContainer, typename Seq, typename Proj>
AssignableContainer& assign_by(AssignableContainer& c, const Seq& seq, const Proj& proj) {
    assign(c, transform_view(seq, proj));
    return c;
}

// Sort in-place
// Note that a const range reference may wrap non-const iterators.

template <typename Seq>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
sort(Seq& seq) {
    auto canon = canonical_view(seq);
    std::sort(std::begin(canon), std::end(canon));
}

template <typename Seq>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
sort(const Seq& seq) {
    auto canon = canonical_view(seq);
    std::sort(std::begin(canon), std::end(canon));
}

// Sort in-place by projection `proj`

template <typename Seq, typename Proj>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
sort_by(Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    std::sort(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

template <typename Seq, typename Proj>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
sort_by(const Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    std::sort(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

// Stable sort in-place by projection `proj`

template <typename Seq, typename Proj>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
stable_sort_by(Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    std::stable_sort(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

template <typename Seq, typename Proj>
enable_if_t<!std::is_const<typename sequence_traits<Seq>::reference>::value>
stable_sort_by(const Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    std::stable_sort(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

// Range-interface for `all_of`, `any_of`

template <typename Seq, typename Predicate>
bool all_of(const Seq& seq, const Predicate& pred) {
    auto canon = canonical_view(seq);
    return std::all_of(std::begin(canon), std::end(canon), pred);
}

template <typename Seq, typename Predicate>
bool any_of(const Seq& seq, const Predicate& pred) {
    auto canon = canonical_view(seq);
    return std::any_of(std::begin(canon), std::end(canon), pred);
}

// Accumulate by projection `proj`

template <
    typename Seq,
    typename Proj,
    typename Value = typename transform_iterator<typename sequence_traits<Seq>::const_iterator, Proj>::value_type
>
Value sum_by(const Seq& seq, const Proj& proj, Value base = Value{}) {
    auto canon = canonical_view(transform_view(seq, proj));
    return std::accumulate(std::begin(canon), std::end(canon), base);
}

// Maximum element by projection `proj`
// - returns an iterator `i` into supplied sequence which has the maximum
//   value of `proj(*i)`.

template <typename Seq, typename Proj>
typename sequence_traits<Seq>::iterator
max_element_by(Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    return std::max_element(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

template <typename Seq, typename Proj>
typename sequence_traits<Seq>::iterator
max_element_by(const Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    return std::max_element(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

// Maximum value.
//
// Value semantics instead of iterator semantics means it will operate
// with input iterators.  Will return default-constructed value if sequence
// is empty.
//
// (Consider making generic associative reduction with TBB implementation
// for random-access iterators?)

template <
    typename Seq,
    typename Value = typename sequence_traits<Seq>::value_type,
    typename Compare = std::less<Value>
>
Value max_value(const Seq& seq, Compare cmp = Compare{}) {
    if (util::empty(seq)) {
        return Value{};
    }

    auto i = std::begin(seq);
    auto e = std::end(seq);
    Value m = *i;
    while (++i!=e) {
        Value x = *i;
        if (cmp(m, x)) {
            m = std::move(x);
        }
    }
    return m;
}

// Minimum and maximum value.

template <
    typename Seq,
    typename Value = typename sequence_traits<Seq>::value_type,
    typename Compare = std::less<Value>
>
std::pair<Value, Value> minmax_value(const Seq& seq, Compare cmp = Compare{}) {
    if (util::empty(seq)) {
        return {Value{}, Value{}};
    }

    auto i = std::begin(seq);
    auto e = std::end(seq);
    Value lower = *i;
    Value upper = *i;
    while (++i!=e) {
        Value x = *i;
        if (cmp(upper, x)) {
            upper = std::move(x);
        }
        else if (cmp(x, lower)) {
            lower = std::move(x);
        }
    }
    return {lower, upper};
}

// View over the keys in an associative container.

template <typename Map>
auto keys(Map& m) DEDUCED_RETURN_TYPE(util::transform_view(m, util::first));

// Test if sequence is sorted after apply projection `proj` to elements.
// (TODO: this will perform unnecessary copies if `proj` returns a reference;
// specialize on this if it becomes an issue.)

template <
    typename Seq,
    typename Proj,
    typename Compare = std::less<typename std::result_of<Proj (typename sequence_traits<Seq>::value_type)>::type>
>
bool is_sorted_by(const Seq& seq, const Proj& proj, Compare cmp = Compare{}) {
    auto i = std::begin(seq);
    auto e = std::end(seq);

    if (i==e) {
        return true;
    }

    // Special one-element case for forward iterators.
    if (is_forward_iterator<decltype(i)>::value) {
        auto j = i;
        if (++j==e) {
            return true;
        }
    }

    auto v = proj(*i++);

    for (;;) {
        if (i==e) {
            return true;
        }
        auto u = proj(*i++);

        if (cmp(u, v)) {
            return false;
        }

        if (i==e) {
            return true;
        }
        v = proj(*i++);

        if (cmp(v, u)) {
            return false;
        }
    }
}

template <typename C, typename Seq>
C make_copy(Seq const& seq) {
    return C{std::begin(seq), std::end(seq)};
}

} // namespace util
} // namespace arb

