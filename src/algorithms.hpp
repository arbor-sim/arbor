#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>

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

    template <typename C>
    bool is_minimal_degree(C const& c)
    {
        static_assert(
            std::is_integral<typename C::value_type>::value,
            "is_minimal_degree only applies to integral types"
        );
        for(auto i=0; i<c.size(); ++i) {
            if(i<c[i]) {
                return false;
            }
        }
        return true;
    }

    template <typename C>
    bool is_positive(C const& c)
    {
        static_assert(
            std::is_integral<typename C::value_type>::value,
            "is_positive only applies to integral types"
        );
        for(auto i=0; i<c.size(); ++i) {
            if(c[i]<1) {
                return false;
            }
        }
        return true;
    }

    template<typename C>
    bool is_contiguously_numbered(const C &parent_list)
    {
        static_assert(
            std::is_integral<typename C::value_type>::value,
            "integral type required"
        );

        std::vector<bool> is_leaf(parent_list.size(), false);

        auto ret = true;
        for (std::size_t i = 1; i < parent_list.size(); ++i) {
            if (is_leaf[parent_list[i]]) {
                ret = false;
                break;
            }

            if (parent_list[i] != i-1) {
                // we have a branch and i-1 is a leaf node
                is_leaf[i-1] = true;
            }
        }

        return ret;
    }

} // namespace algorithms
} // namespace mc
} // namespace nest
