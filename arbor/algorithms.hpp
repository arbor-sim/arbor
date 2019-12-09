#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include <arbor/assert.hpp>

#include <arbor/util/compat.hpp>
#include <util/meta.hpp>
#include <util/range.hpp>
#include <util/rangeutil.hpp>

/*
 * Some simple wrappers around stl algorithms to improve readability of code
 * that uses them.
 *
 * For example, a simple sum() wrapper for finding the sum of a container,
 * is much more readable than using std::accumulate()
 *
 */

namespace arb {
namespace algorithms {

template <typename C>
typename util::sequence_traits<C>::value_type
mean(C const& c)
{
    return util::sum(c)/util::size(c);
}

// returns the prefix sum of c in the form `[0, c[0], c[0]+c[1], ..., sum(c)]`.
// This means that the returned vector has one more element than c.
template <typename C>
C make_index(C const& c)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "make_index only applies to integral types"
    );

    C out(c.size()+1);
    out[0] = 0;
    std::partial_sum(c.begin(), c.end(), out.begin()+1);
    return out;
}

/// test for membership within half-open interval
template <typename X, typename I, typename J>
bool in_interval(const X& x, const I& lower, const J& upper) {
    return x>=lower && x<upper;
}

template <typename X, typename I, typename J>
bool in_interval(const X& x, const std::pair<I, J>& bounds) {
    return x>=bounds.first && x<bounds.second;
}

/// works like std::is_sorted(), but with stronger condition that succesive
/// elements must be greater than those before them
template <typename C>
bool is_strictly_monotonic_increasing(C const& c)
{
    using value_type = typename C::value_type;
    return std::is_sorted(
        c.begin(),
        c.end(),
        [] (value_type const& lhs, value_type const& rhs) {
            return lhs <= rhs;
        }
    );
}

template <typename C>
bool is_strictly_monotonic_decreasing(C const& c)
{
    using value_type = typename C::value_type;
    return std::is_sorted(
        c.begin(),
        c.end(),
        [] (value_type const& lhs, value_type const& rhs) {
            return lhs >= rhs;
        }
    );
}

// Check if c[0] == 0 and c[i] < 0 holds for i != 0
// Also handle the valid case of c[0]==value_type(-1)
// This means that children of a node always have larger indices than their
// parent.
template <
    typename C,
    typename = typename std::enable_if<std::is_integral<typename C::value_type>::value>
>
bool is_minimal_degree(C const& c)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "is_minimal_degree only applies to integral types"
    );

    if (c.size()==0u) {
        return true;
    }

    using value_type = typename C::value_type;
    if (!(c[0]==value_type(0) || c[0]==value_type(-1))) {
        return false;
    }
    auto i = value_type(1);
    auto it = std::find_if(
        c.begin()+1, c.end(), [&i](value_type v) { return v>=(i++); }
    );
    return it==c.end();
}

template <typename C>
bool all_positive(const C& c) {
    return util::all_of(c, [](auto v) { return v>decltype(v){}; });
}

template <typename C>
bool all_negative(const C& c) {
    return util::all_of(c, [](auto v) { return v<decltype(v){}; });
}

// returns a vector containing the number of children for each node.
template<typename C>
std::vector<typename C::value_type> child_count(const C& parent_index)
{
    using value_type = typename C::value_type;
    static_assert(
        std::is_integral<value_type>::value,
        "integral type required"
    );

    std::vector<value_type> count(parent_index.size(), 0);
    for (auto i = 0u; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        // -1 means no parent
        if (p != value_type(i) && p != value_type(-1)) {
            ++count[p];
        }
    }

    return count;
}

template<typename C>
bool has_contiguous_compartments(const C& parent_index)
{
    using value_type = typename C::value_type;
    static_assert(
        std::is_integral<value_type>::value,
        "integral type required"
    );

    if (!is_minimal_degree(parent_index)) {
        return false;
    }

    auto num_child = child_count(parent_index);
    for (auto i=1u; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        if (num_child[p]==1 && p!=value_type(i-1)) {
            return false;
        }
    }

    return true;
}

template<typename C>
std::vector<typename C::value_type> branches(const C& parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    arb_assert(has_contiguous_compartments(parent_index));

    std::vector<typename C::value_type> branch_index;
    if (parent_index.empty()) {
        return branch_index;
    }

    auto num_child = child_count(parent_index);
    branch_index.push_back(0);
    for (std::size_t i = 1; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        if (num_child[p] > 1 || parent_index[p] == p) {
            // `parent_index[p] == p` ~> parent_index[i] is the soma
            branch_index.push_back(i);
        }
    }

    branch_index.push_back(parent_index.size());
    return branch_index;
}

// creates a vector that contains the branch index for each compartment.
// e.g. {0, 1, 5, 9, 10} -> {0, 1, 1, 1, 1, 2, 2, 2, 2, 3}
//                  indices  0  1           5           9
template<typename C>
std::vector<typename C::value_type> expand_branches(const C& branch_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (branch_index.empty())
        return {};

    std::vector<typename C::value_type> expanded(branch_index.back());
    for (std::size_t i = 0; i < branch_index.size()-1; ++i) {
        for (auto j = branch_index[i]; j < branch_index[i+1]; ++j) {
            expanded[j] = i;
        }
    }

    return expanded;
}

template<typename C>
typename C::value_type find_branch(const C& branch_index,
                                   typename C::value_type nid)
{
    using value_type = typename C::value_type;
    static_assert(
        std::is_integral<value_type>::value,
        "integral type required"
    );

    auto it =  std::find_if(
        branch_index.begin(), branch_index.end(),
        [nid](const value_type& v) { return v > nid; }
    );

    return it - branch_index.begin() - 1;
}

// Find the reduced form of a tree represented as a parent index.
// The operation of transforming a tree into a homeomorphic pre-image is called
// 'reduction'. The 'reduced' tree is the smallest tree that maps into the tree
// homeomorphically (i.e. preserving labels and child relationships).
// For example, the tree represented by the following index:
//      {0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12}
// Has reduced form
//      {0, 0, 0, 0, 3, 3}
//
// This transformation can be represented graphically:
//
//        0               0
//       /|\             /|\.
//      1 4 6      =>   1 2 3
//     /  |  \             / \.
//    2   5   7           4   5
//   /         \.
//  3           8
//             / \.
//            9   11
//           /     \.
//          10     12
//                   \.
//                   13
//
template<typename C>
std::vector<typename C::value_type> tree_reduce(
    const C& parent_index, const C& branch_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (parent_index.empty() && branch_index.empty()) {
        return {};
    }

    arb_assert(parent_index.size()-branch_index.back() == 0);
    arb_assert(has_contiguous_compartments(parent_index));
    arb_assert(is_strictly_monotonic_increasing(branch_index));

    // expand the branch index to lookup the banch id for each compartment
    auto expanded_branch = expand_branches(branch_index);

    std::vector<typename C::value_type> new_parent_index;
    // push the first element manually as the parent of the root might be -1
    new_parent_index.push_back(expanded_branch[0]);
    for (std::size_t i = 1; i < branch_index.size()-1; ++i) {
        auto p = parent_index[branch_index[i]];
        new_parent_index.push_back(expanded_branch[p]);
    }

    return new_parent_index;
}

template <typename Seq, typename = util::enable_if_sequence_t<Seq&>>
bool is_unique(const Seq& seq) {
    using std::begin;
    using std::end;

    return std::adjacent_find(begin(seq), end(seq)) == end(seq);
}


/// Binary search, because std::binary_search doesn't return information
/// about where a match was found.

// TODO: consolidate these with rangeutil routines; make them sentinel

template <typename It, typename T>
It binary_find(It b, It e, const T& value) {
    auto it = std::lower_bound(b, e, value);
    return it==e ? e : (*it==value ? it : e);
}

template <typename Seq, typename T>
auto binary_find(const Seq& seq, const T& value) {
    using std::begin;
    using std::end;

    return binary_find(begin(seq), end(seq), value);
}

template <typename Seq, typename T>
auto binary_find(Seq& seq, const T& value) {
    using std::begin;
    using std::end;

    return binary_find(begin(seq), end(seq), value);
}

} // namespace algorithms
} // namespace arb
