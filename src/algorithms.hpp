#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include "util.hpp"
#include "util/debug.hpp"

/*
 * Some simple wrappers around stl algorithms to improve readability of code
 * that uses them.
 *
 * For example, a simple sum() wrapper for finding the sum of a container,
 * is much more readable than using std::accumulate()
 *
 */

namespace nest {
namespace mc {
namespace algorithms {

template <typename C>
typename C::value_type
sum(C const& c)
{
    using value_type = typename C::value_type;
    return std::accumulate(c.begin(), c.end(), value_type{0});
}

template <typename C>
typename C::value_type
mean(C const& c)
{
    return sum(c)/c.size();
}

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

    if(c.size()==0u) {
        return true;
    }

    using value_type = typename C::value_type;
    if(c[0] != value_type(0)) {
        return false;
    }
    auto i = value_type(1);
    auto it = std::find_if(
        c.begin()+1, c.end(), [&i](value_type v) { return v>=(i++); }
    );
    return it==c.end();
}

template <typename C>
bool is_positive(C const& c)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "is_positive only applies to integral types"
    );
    for(auto v : c) {
        if(v<1) {
            return false;
        }
    }
    return true;
}

template<typename C>
bool has_contiguous_segments(const C &parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (!is_minimal_degree(parent_index)) {
        return false;
    }

    int n = parent_index.size();
    std::vector<bool> is_leaf(n, false);

    for(auto i=1; i<n; ++i) {
        auto p = parent_index[i];
        if(is_leaf[p]) {
            return false;
        }

        if(p != i-1) {
            // we have a branch and i-1 is a leaf node
            is_leaf[i-1] = true;
        }
    }

    return true;
}

template<typename C>
std::vector<typename C::value_type> child_count(const C &parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    std::vector<typename C::value_type> count(parent_index.size(), 0);
    for (std::size_t i = 1; i < parent_index.size(); ++i) {
        ++count[parent_index[i]];
    }

    return count;
}

template<typename C>
std::vector<typename C::value_type> branches(const C& parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    EXPECTS(has_contiguous_segments(parent_index));

    std::vector<typename C::value_type> branch_index;
    if (parent_index.empty()) {
        return branch_index;
    }

    auto num_child = child_count(parent_index);
    branch_index.push_back(0);
    for (std::size_t i = 1; i < parent_index.size(); ++i) {
        auto p = parent_index[i];
        if (num_child[p] > 1 || parent_index[p] == p) {
            // parent_index[p] == p -> parent_index[i] is the soma
            branch_index.push_back(i);
        }
    }

    branch_index.push_back(parent_index.size());
    return branch_index;
}


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
        for (std::size_t j = branch_index[i]; j < branch_index[i+1]; ++j) {
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
        [nid](const value_type &v) { return v > nid; }
    );

    return it - branch_index.begin() - 1;
}


template<typename C>
std::vector<typename C::value_type> make_parent_index(
    const C& parent_index, const C& branch_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    if (parent_index.empty() && branch_index.empty()) {
        return {};
    }

    EXPECTS(parent_index.size() == branch_index.back());
    EXPECTS(has_contiguous_segments(parent_index));
    EXPECTS(is_strictly_monotonic_increasing(branch_index));

    // expand the branch index
    auto expanded_branch = expand_branches(branch_index);

    std::vector<typename C::value_type> new_parent_index;
    for (std::size_t i = 0; i < branch_index.size()-1; ++i) {
        auto p = parent_index[branch_index[i]];
        new_parent_index.push_back(expanded_branch[p]);
    }

    return new_parent_index;
}


template<typename C>
bool is_sorted(const C& c)
{
    return std::is_sorted(c.begin(), c.end());
}

template<typename C>
bool is_unique(const C& c)
{
    return std::adjacent_find(c.begin(), c.end()) == c.end();
}

/// Return and index that maps entries in sub to their corresponding
/// values in super, where sub is a subset of super.
///
/// Both sets are sorted and have unique entries.
/// Complexity is O(n), where n is size of super
template<typename C>
// C::iterator models forward_iterator
// C::value_type is_integral
C index_into(const C& super, const C& sub)
{
    //EXPECTS {s \in super : \forall s \in sub};
    EXPECTS(is_unique(super) && is_unique(sub));
    EXPECTS(is_sorted(super) && is_sorted(sub));
    EXPECTS(sub.size() <= super.size());

    static_assert(
        std::is_integral<typename C::value_type>::value,
        "index_into only applies to integral types"
    );

    C out(sub.size()); // out will have one entry for each index in sub

    auto sub_it=sub.begin();
    auto super_it=super.begin();
    auto sub_idx=0u, super_idx = 0u;

    while(sub_it!=sub.end() && super_it!=super.end()) {
        if(*sub_it==*super_it) {
            out[sub_idx] = super_idx;
            ++sub_it; ++sub_idx;
        }
        ++super_it; ++super_idx;
    }

    EXPECTS(sub_idx==sub.size());

    return out;
}

} // namespace algorithms
} // namespace mc
} // namespace nest
