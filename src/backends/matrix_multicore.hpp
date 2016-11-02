#pragma once

#include <util/span.hpp>
#include "memory_traits.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct matrix_solver: public memory_traits {
    void solve(
        view d, view u, view rhs,
        const_iview p, const_iview cell_index)
    {
        const size_type ncells = cell_index.size()-1;

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto first = cell_index[m];
            auto last = cell_index[m+1];

            // backward sweep
            for(auto i=last-1; i>first; --i) {
                auto factor = u[i] / d[i];
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

struct fvm_matrix_builder: public memory_traits {
    view d;
    view u;
    view rhs;
    const_iview p;

    const_view sigma;
    const_view alpha;
    const_view voltage;
    const_view current;
    const_view cv_capacitance;

    array alpha_d;

    fvm_matrix_builder() = default;

    fvm_matrix_builder(
        view d, view u, view rhs, const_iview p,
        const_view sigma, const_view alpha,
        const_view voltage, const_view current, const_view cv_capacitance)
    :
        d{d}, u{u}, rhs{rhs}, p{p},
        sigma{sigma}, alpha{alpha},
        voltage{voltage}, current{current}, cv_capacitance{cv_capacitance}
    {
        auto n = d.size();
        alpha_d = array(n, 0);
        for(auto i: util::make_span(1u, n)) {
            alpha_d[i] += alpha[i];

            // add contribution to the diagonal of parent
            alpha_d[p[i]] += alpha[i];
        }
    }

    void build(value_type dt) {
        auto n = d.size();
        value_type factor_lhs = 1e5*dt;
        value_type factor_rhs = 1e1*dt; //  units: 10·ms/(F/m^2)·(mA/cm^2) ≡ mV
        for (auto i: util::make_span(0u, n)) {
            d[i] = sigma[i] + factor_lhs*alpha_d[i];
            u[i] = -factor_lhs*alpha[i];
            // the RHS of the linear system is
            //      cv_area * (V - dt/cm*(im - ie))
            rhs[i] = sigma[i]*(voltage[i] - factor_rhs/cv_capacitance[i]*current[i]);
        }
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
