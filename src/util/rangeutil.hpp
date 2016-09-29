#pragma once

/*
 * Sequence and container utilities compatible
 * with ranges.
 */

#include <iterator>

#include <util/meta.hpp>
#include <util/range.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {
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

// Assign sequence to a container with transform `proj`

template <typename AssignableContainer, typename Seq, typename Proj>
AssignableContainer& assign_by(AssignableContainer& c, const Seq& seq, const Proj& proj) {
    auto canon = canonical_view(transform_view(seq, proj));
    c.assign(std::begin(canon), std::end(canon));
    return c;
}

// Sort in-place

template <typename Seq>
void sort(Seq& seq) {
    auto canon = canonical_view(seq);
    std::sort(std::begin(canon), std::end(canon));
}

// Sort in-place by projection `proj`

template <typename Seq, typename Proj>
void sort_by(Seq& seq, const Proj& proj) {
    using value_type = typename sequence_traits<Seq>::value_type;
    auto canon = canonical_view(seq);

    std::sort(std::begin(canon), std::end(canon),
        [&proj](const value_type& a, const value_type& b) {
            return proj(a) < proj(b);
        });
}

} // namespace util
} // namespace mc
} // namespace nest
