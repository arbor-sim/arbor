#pragma once

/*
 * Sequence and container utilities compatible
 * with ranges.
 */

#include <algorithm>
#include <iterator>
#include <numeric>
#include <cstring>
#include <type_traits>
#include <ranges>

#include "util/meta.hpp"
#include "util/range.hpp"

namespace arb {
namespace util {

namespace detail {
    // Type acts as a tag to find the correct operator| overload
    template <typename C>
    struct to_helper {};

    // This actually does the work
    template <typename Container, std::ranges::range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, typename Container::value_type>
    Container operator|(R&& r, to_helper<Container>) { return Container{r.begin(), r.end()}; }
}

template <typename Container>
auto to() { return detail::to_helper<Container>{}; }

// Non-owning views and subviews
template <typename Seq>
range<typename sequence_traits<Seq&&>::iterator, typename sequence_traits<Seq&&>::sentinel>
range_view(Seq&& seq) { return make_range(std::begin(seq), std::end(seq)); }

template <typename Seq, typename = std::enable_if_t<sequence_traits<Seq&&>::is_contiguous>>
auto range_pointer_view(Seq&& seq) { return make_range(std::data(seq), std::data(seq)+std::size(seq)); }

template <typename Seq,
          typename Offset1,
          typename Offset2,
          typename Iter = typename sequence_traits<Seq&&>::iterator>
std::enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq&& seq, Offset1 bi, Offset2 ei) {
    Iter b = std::next(std::begin(seq), bi);
    Iter e = std::next(b, ei - bi);
    return make_range(b, e);
}

template <typename Seq,
          typename Offset1,
          typename Offset2,
          typename Iter = typename sequence_traits<Seq&&>::iterator>
std::enable_if_t<is_forward_iterator<Iter>::value, range<Iter>>
subrange_view(Seq&& seq, std::pair<Offset1, Offset2> index) {
    return subrange_view(std::forward<Seq>(seq), index.first, index.second);
}

// helper for determining the type of a subrange_view
template <typename Seq>
using subrange_view_type = decltype(subrange_view(std::declval<Seq&>(), 0, 0));

// Zero a container, specialised for contiguous sequences
// i.e.: Array, Vector, String.
template <typename Seq,
          typename = std::enable_if_t<std::is_trivially_copyable_v<typename sequence_traits<Seq&&>::value_type>
                                   && sequence_traits<Seq&&>::is_contiguous>>
void zero(Seq& vs) {
    // NOTE: All contiguous containers have `data` and `size` methods.
    using T = typename sequence_traits<Seq&&>::value_type;
    std::memset(vs.data(), 0x0, vs.size()*sizeof(T));
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

// Assign sequence to a container with transform `proj`
template <typename AssignableContainer, typename Seq, typename Proj>
AssignableContainer& assign_by(AssignableContainer& c, const Seq& seq, const Proj& proj) {
    assign(c, seq | std::ranges::views::transform(proj));
    return c;
}

// Sort in-place by projection `proj`
template <typename Seq, typename Proj>
std::enable_if_t<!std::is_const<typename sequence_traits<Seq&&>::reference>::value>
sort_by(Seq&& seq, const Proj& proj) { std::ranges::sort(seq, std::ranges::less{}, proj); }

// Accumulate over range
template <
    typename Seq,
    typename Value = typename util::sequence_traits<Seq>::value_type
>
Value sum(const Seq& seq, Value base = Value{}) {
    auto canon = canonical_view(seq);
    return std::accumulate(canon.begin(), canon.end(), base);
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

