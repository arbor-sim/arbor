#pragma once

/*
 * Sequence and container utilities compatible
 * with ranges.
 */

#include <algorithm>
#include <iterator>
#include <ostream>
#include <numeric>

#include "util/meta.hpp"
#include "util/range.hpp"
#include "util/transform.hpp"

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
range<typename sequence_traits<Seq&&>::iterator, typename sequence_traits<Seq&&>::sentinel>
range_view(Seq&& seq) {
    return make_range(std::begin(seq), std::end(seq));
}

template <typename Seq, typename = std::enable_if_t<sequence_traits<Seq&&>::is_contiguous>>
auto range_pointer_view(Seq&& seq) {
    return make_range(std::data(seq), std::data(seq)+std::size(seq));
}

template <
    typename Seq,
    typename Offset1,
    typename Offset2,
    typename Iter = typename sequence_traits<Seq&&>::iterator
>
std::enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq&& seq, Offset1 bi, Offset2 ei) {
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
    typename Iter = typename sequence_traits<Seq&&>::iterator
>
std::enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq&& seq, std::pair<Offset1, Offset2> index) {
    return subrange_view(std::forward<Seq>(seq), index.first, index.second);
}

// helper for determining the type of a subrange_view
template <typename Seq>
using subrange_view_type = decltype(subrange_view(std::declval<Seq&>(), 0, 0));


// Fill container or range.

template <typename Seq, typename V>
void fill(Seq&& seq, const V& value) {
    auto canon = canonical_view(seq);
    std::fill(canon.begin(), canon.end(), value);
}

// Append sequence to a container

template <typename Container, typename Seq>
Container& append(Container &c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.insert(c.end(), canon.begin(), canon.end());
    return c;
}

// Assign sequence to a container

template <typename AssignableContainer, typename Seq>
AssignableContainer& assign(AssignableContainer& c, const Seq& seq) {
    auto canon = canonical_view(seq);
    c.assign(canon.begin(), canon.end());
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
            auto canon = canonical_view(ref);
            return C(canon.begin(), canon.end());
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
std::enable_if_t<!std::is_const<typename sequence_traits<Seq&&>::reference>::value>
sort(Seq&& seq) {
    auto canon = canonical_view(seq);
    std::sort(canon.begin(), canon.end());
}

template <typename Seq, typename Less>
std::enable_if_t<!std::is_const<typename sequence_traits<Seq&&>::reference>::value>
sort(Seq&& seq, const Less& less) {
    auto canon = canonical_view(seq);
    std::sort(canon.begin(), canon.end(), less);
}

// Sort in-place by projection `proj`

template <typename Seq, typename Proj>
std::enable_if_t<!std::is_const<typename sequence_traits<Seq&&>::reference>::value>
sort_by(Seq&& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq&&>::value_type;
    auto canon = canonical_view(seq);

    std::sort(canon.begin(), canon.end(),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

// Stable sort in-place by projection `proj`

template <typename Seq, typename Proj>
std::enable_if_t<!std::is_const<typename sequence_traits<Seq&&>::reference>::value>
stable_sort_by(Seq&& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq&&>::value_type;
    auto canon = canonical_view(seq);

    std::stable_sort(canon.begin(), canon.end(),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

// Range-interface for `all_of`, `any_of`

template <typename Seq, typename Predicate>
bool all_of(const Seq& seq, const Predicate& pred) {
    auto canon = canonical_view(seq);
    return std::all_of(canon.begin(), canon.end(), pred);
}

template <typename Seq, typename Predicate>
bool any_of(const Seq& seq, const Predicate& pred) {
    auto canon = canonical_view(seq);
    return std::any_of(canon.begin(), canon.end(), pred);
}

// Accumulate over range

template <
    typename Seq,
    typename Value = typename util::sequence_traits<Seq>::value_type
>
Value sum(const Seq& seq, Value base = Value{}) {
    auto canon = canonical_view(seq);
    return std::accumulate(canon.begin(), canon.end(), base);
}

// Accumulate by projection `proj`

template <
    typename Seq,
    typename Proj,
    typename Value = typename transform_iterator<typename sequence_traits<const Seq&>::const_iterator, Proj>::value_type
>
Value sum_by(const Seq& seq, const Proj& proj, Value base = Value{}) {
    auto canon = canonical_view(transform_view(seq, proj));
    return std::accumulate(canon.begin(), canon.end(), base);
}

// Maximum element by projection `proj`
// - returns an iterator `i` into supplied sequence which has the maximum
//   value of `proj(*i)`.

template <typename Seq, typename Proj>
typename sequence_traits<Seq&&>::iterator
max_element_by(Seq&& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq&&>::value_type;
    auto canon = canonical_view(seq);

    return std::max_element(canon.begin(), canon.end(),
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
    typename Value = typename sequence_traits<const Seq&>::value_type,
    typename Compare = std::less<Value>
>
Value max_value(const Seq& seq, Compare cmp = Compare{}) {
    using std::begin;
    using std::end;

    if (std::empty(seq)) {
        return Value{};
    }

    auto i = begin(seq);
    auto e = end(seq);
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
    typename Value = typename sequence_traits<const Seq&>::value_type,
    typename Compare = std::less<Value>
>
std::pair<Value, Value> minmax_value(const Seq& seq, Compare cmp = Compare{}) {
    using std::begin;
    using std::end;

    if (std::empty(seq)) {
        return {Value{}, Value{}};
    }

    auto i = begin(seq);
    auto e = end(seq);
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

// Range-wrapper for std::is_sorted.

template <typename Seq, typename = util::enable_if_sequence_t<const Seq&>>
bool is_sorted(const Seq& seq) {
    auto canon = canonical_view(seq);
    return std::is_sorted(canon.begin(), canon.end());
}

// Range version of std::equal.

template <
    typename Seq1, typename Seq2,
    typename Eq = std::equal_to<void>,
    typename = util::enable_if_sequence_t<const Seq1&>,
    typename = util::enable_if_sequence_t<const Seq2&>
>
bool equal(const Seq1& seq1, const Seq2& seq2, Eq p = Eq{}) {
    using std::begin;
    using std::end;

    auto i1 = begin(seq1);
    auto e1 = end(seq1);

    auto i2 = begin(seq2);
    auto e2 = end(seq2);

    while (i1!=e1 && i2!=e2) {
        if (!p(*i1, *i2)) return false;
        ++i1;
        ++i2;
    }
    return i1==e1 && i2==e2;
}

// Test if sequence is sorted after apply projection `proj` to elements.
// (TODO: this will perform unnecessary copies if `proj` returns a reference;
// specialize on this if it becomes an issue.)

template <
    typename Seq,
    typename Proj,
    typename Compare = std::less<std::result_of_t<Proj (typename sequence_traits<const Seq&>::value_type)>>
>
bool is_sorted_by(const Seq& seq, const Proj& proj, Compare cmp = Compare{}) {
    using std::begin;
    using std::end;

    auto i = begin(seq);
    auto e = end(seq);

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

// Present a view of a finite sequence in reverse order, provided
// the sequence iterator is bidirectional.
template <
    typename Seq,
    typename It = typename sequence_traits<Seq&&>::iterator,
    typename Rev = std::reverse_iterator<It>
>
range<Rev, Rev> reverse_view(Seq&& seq) {
    auto strict = strict_view(seq);
    return range<Rev, Rev>(Rev(strict.right), Rev(strict.left));
}

// Left fold (accumulate) over sequence.
//
// Note that the order of arguments follows the application order;
// schematically:
//
//     foldl f a [] = a
//     foldl f a (b:bs) = foldl f (f a b) bs
//
// The binary operator f will be invoked once per element in the
// sequence in turn, with the running accumulator as the first
// argument. If the iterators for the sequence deference to a
// mutable lvalue, then mutation of the value in the input sequence
// is explicitly permitted.
//
// std::accumulate(begin, end, init, f) is equivalent to
// util::foldl(f, init, util::make_range(begin, end)).

template <typename Seq, typename Acc, typename BinOp>
auto foldl(BinOp f, Acc a, Seq&& seq) {
    using std::begin;
    using std::end;

    auto b = begin(seq);
    auto e = end(seq);

    while (b!=e) {
        a = f(std::move(a), *b);
        ++b;
    }
    return a;
}

// Copy elements from source sequence into destination sequence,
// and fill the remaining elements of the destination sequence
// with the given fill value.
//
// Assumes that the iterators for these sequences are at least
// forward iterators.
template <typename Source, typename Dest, typename Fill>
void copy_extend(const Source& source, Dest&& dest, const Fill& fill) {
    using std::begin;
    using std::end;

    auto dest_n = std::size(dest);
    auto source_n = std::size(source);

    auto n = source_n<dest_n? source_n: dest_n;
    auto tail = std::copy_n(begin(source), n, begin(dest));
    std::fill(tail, end(dest), fill);
}

} // namespace util
} // namespace arb

