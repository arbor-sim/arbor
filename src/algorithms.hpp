#pragma once

#include <iostream>

#include <algorithm>
#include <numeric>
#include <type_traits>

#include "util.hpp"

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
namespace algorithms{

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
