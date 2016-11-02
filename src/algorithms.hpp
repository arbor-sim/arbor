#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include <util/debug.hpp>
#include <util/range.hpp>

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
    if (c[0] != value_type(0)) {
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
std::vector<typename C::value_type> child_count(const C& parent_index)
{
    static_assert(
        std::is_integral<typename C::value_type>::value,
        "integral type required"
    );

    std::vector<typename C::value_type> count(parent_index.size(), 0);
    for (auto i = 1u; i < parent_index.size(); ++i) {
        ++count[parent_index[i]];
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

    EXPECTS(has_contiguous_compartments(parent_index));

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

    EXPECTS(parent_index.size()-branch_index.back() == 0);
    EXPECTS(has_contiguous_compartments(parent_index));
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

template <typename R1, typename R2>
class index_into_iterator:
    public std::iterator<std::forward_iterator_tag, typename R1::value_type>
{
public:
    using value_type = typename R1::value_type;

private:
    using super_iterator = typename R1::const_iterator;
    using sub_iterator = typename R2::const_iterator;

    mutable super_iterator super_it_;
    const super_iterator super_end_;

    sub_iterator sub_it_;

    mutable value_type super_idx_;

public:
    index_into_iterator(super_iterator sup, super_iterator sup_end, sub_iterator sub) :
        super_it_(sup), super_end_(sup_end),
        sub_it_(sub), super_idx_(0)
    {}

    index_into_iterator(const R1& super, const R2& sub):
        super_it_  (std::begin(super)),
        super_end_ (std::end(super)),
        sub_it_    (std::begin(sub)),
        super_idx_ (0)
    {}

    value_type operator* () {
        advance_super();
        return super_idx_;
    }

    value_type operator* () const {
        advance_super();
        return super_idx_;
    }

    bool operator== (const index_into_iterator& other) {
        return sub_it_ == other.sub_it_;
    }

    bool operator!= (const index_into_iterator& other) {
        return !(*this == other);
    }

    index_into_iterator operator++ () {
        ++sub_it_;
        return (*this);
    }

    index_into_iterator operator++(int) {
        auto previous = *this;
        ++(*this);
        return previous;
    }

private:

    bool is_aligned() const {
        return *sub_it_ == *super_it_;
    }

    void advance_super() {
        while(super_it_!=super_end_ && !is_aligned()) {
            ++super_it_;
            ++super_idx_;
        }

        // this indicates that no match was found in super for a value
        // in sub, which violates the precondition that sub is a subset of super
        EXPECTS(!(super_it_==super_end_));
    }
};

/// Return and index that maps entries in sub to their corresponding
/// values in super, where sub is a subset of super.
///
/// Both sets are sorted and have unique entries.
/// Complexity is O(n), where n is size of super
template<typename R1, typename R2>
util::range<index_into_iterator<R1, R2>>
index_into(const R1& super, const R2& sub)
{
    EXPECTS(is_unique(super) && is_unique(sub));
    EXPECTS(is_sorted(super) && is_sorted(sub));
    EXPECTS(sub.size() <= super.size());

    using iterator = index_into_iterator<R1, R2>;
    auto begin = iterator(super, sub);
    auto end   = iterator(super.end(), super.end(), sub.end());
    return util::make_range(begin, end);
}

} // namespace algorithms
} // namespace mc
} // namespace nest
