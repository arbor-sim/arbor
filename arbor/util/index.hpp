#pragma once

// Implementation of prefix sum

#include <numeric>

namespace arb {
namespace util {

// Returns the prefix sum of c in the form `[0, c[0], c[0]+c[1], ..., sum(c)]`.
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

} // namespace util
} // namespace arb
