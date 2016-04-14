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

        using value_type = typename C::value_type;
        auto i = value_type(0);
        for(auto v : c) {
            if(i++<v) {
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
        for(auto v : c) {
            if(v<1) {
                return false;
            }
        }
        return true;
    }

} // namespace algorithms
} // namespace mc
} // namespace nest
