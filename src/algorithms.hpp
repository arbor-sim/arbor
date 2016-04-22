#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

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

        std::vector<bool> is_leaf(parent_index.size(), false);

        for (std::size_t i = 1; i < parent_index.size(); ++i) {
            auto p = parent_index[i];
            if (is_leaf[p]) {
                return false;
            }

            if (p != i-1) {
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

    template<typename C, bool CheckStrict = true>
    std::vector<typename C::value_type> branches(const C &parent_index)
    {
        static_assert(
            std::is_integral<typename C::value_type>::value,
            "integral type required"
        );

        if (CheckStrict && !has_contiguous_segments(parent_index)) {
            throw std::invalid_argument(
                "parent_index has not contiguous branch numbering"
            );
        }

        auto num_child = child_count(parent_index);
        std::vector<typename C::value_type> branch_runs(
            parent_index.size(), 0
        );

        std::size_t num_branches = (num_child[0] == 1) ? 1 : 0;
        for (std::size_t i = 1; i < parent_index.size(); ++i) {
            auto p = parent_index[i];
            if (num_child[p] > 1) {
                ++num_branches;
            }

            branch_runs[i] = num_branches;
        }

        return branch_runs;
    }

    template<typename C>
    std::vector<typename C::value_type> branches_fast(const C &parent_index)
    {
        return branches<C,false>(parent_index);
    }

} // namespace algorithms
} // namespace mc
} // namespace nest
