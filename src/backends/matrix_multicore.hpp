#pragma once

#include "memory_traits.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct matrix_policy : public memory_traits {
    void solve(
        view l, view d, view u, view rhs,
        const_iview p, const_iview cell_index)
    {
        const size_type ncells = cell_index.size()-1;

        // loop over submatrices
        for (size_type m=0; m<ncells; ++m) {
            auto first = cell_index[m];
            auto last = cell_index[m+1];

            // backward sweep
            for(auto i=last-1; i>first; --i) {
                auto factor = l[i] / d[i];
                d[p[i]]   -= factor * u[i];
                rhs[p[i]] -= factor * rhs[i];
            }
            rhs[first] /= d[first];

            // forward sweep
            for(auto i=first+1; i<last; ++i) {
                rhs[i] -= u[i] * rhs[p[i]];
                rhs[i] /= d[i];
            }
        }
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
