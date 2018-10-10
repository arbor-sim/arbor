#pragma once

#include <fvm_layout.hpp>
#include <util/partition.hpp>
#include <util/span.hpp>

#include "multicore_common.hpp"
#include "profile/profiler_macro.hpp"

namespace arb {
namespace multicore {

template <typename T, typename I>
struct matrix_state {
public:
    using value_type = T;
    using index_type = I;

    using array = padded_vector<value_type>;
    using const_view = const array&;

    using iarray = padded_vector<index_type>;
    iarray parent_index;
    iarray cell_cv_divs;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]
    array x;     // solution

    array cv_capacitance;      // [pF]
    array face_conductance;    // [μS]
    array cv_area;             // [μm^2]

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    std::vector<gap_junction> gj;

    array r, p, Ap, MinvR, x_0;
    array scalars;	// array of variables (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

    matrix_state() = default;

    matrix_state(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond,
                 const std::vector<value_type>& area,
                 const std::vector<gap_junction>& gj_coords):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        d(size(), 0), u(size(), 0), rhs(size()), x(size(), 0),
        cv_capacitance(cap.begin(), cap.end()),
        face_conductance(cond.begin(), cond.end()),
        cv_area(area.begin(), area.end()), gj(gj_coords),
        r(size()), p(size()), Ap(size()),
        MinvR(size()), x_0(size(), 0), scalars(3)
    {
        arb_assert(cap.size() == size());
        arb_assert(cond.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());

        auto n = size();
        invariant_d = array(n, 0);
        for (auto i: util::make_span(1u, n)) {
            auto gij = face_conductance[i];

            u[i] = -gij;
            invariant_d[i] += gij;
            invariant_d[p[i]] += gij;
        }
    }

    const_view solution() const {
        // In this back end the solution is a simple view of the rhs, which
        // contains the solution after the matrix_solve is performed.
        return x;
    }


    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
    //   dt_cell         [ms]     (per cell)
    //   voltage         [mV]     (per compartment)
    //   current density [A.m^-2] (per compartment)
    void assemble(const_view dt_cell, const_view voltage, const_view current) {
        auto cell_cv_part = util::partition_view(cell_cv_divs);
        const index_type ncells = cell_cv_part.size();

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto dt = dt_cell[m];

            if (dt>0) {
                value_type factor = 1e-3/dt;
                for (auto i: util::make_span(cell_cv_part[m])) {
                    auto gi = factor*cv_capacitance[i];

                    d[i] = gi + invariant_d[i];
                    // convert current to units nA
                    rhs[i] = gi*voltage[i] - 1e-3*cv_area[i]*current[i];
                }
            }
            else {
                for (auto i: util::make_span(cell_cv_part[m])) {
                    d[i] = 0;
                    rhs[i] = voltage[i];
                }
            }
        }
    }

    void solve_tdma() {
        // loop over submatrices
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            auto first = cv_span.first;
            auto last = cv_span.second; // one past the end

            auto t = d;

            if (t[first]!=0) {
                // backward sweep
                for(auto i=last-1; i>first; --i) {
                    auto factor = u[i] / t[i];
                    t[parent_index[i]] -= factor * u[i];
                    x[parent_index[i]] -= factor * x[i];
                }
                x[first] /= t[first];

                // forward sweep
                for(auto i=first+1; i<last; ++i) {
                    x[i] -= u[i] * x[parent_index[i]];
                    x[i] /= t[i];
                }
            }
        }
    }

    void solve(const array& b, array& c) {
        x = b;
        solve_tdma();
        c = x;
    }

    void solve() {
        x = rhs;
        solve_tdma();
    }

    void MatrixVectorProduct(const array& b, array& c, bool precond) {
        c[0] = d[0] *b[0];
        for (unsigned i = 1; i < size(); i++) {
            unsigned p = parent_index[i];
            c[i] = d[i] * b[i] +  u[i] * b[p];
            c[p] += u[i] * b[i];
        }
        if(!precond) {
            for (auto g: gj) {
                c[g.loc.first] += g.weight * b[g.loc.second];
            }
        }
    };

    void Set_Up_CG() {

        // Reset result vector
        std::fill(x_0.begin(), x_0.end(), 0);

        // Set residual = rhs
        r = rhs;

        // Solve Mx = R; store in MinvR, M being the preconditioning matrix
        solve(r, MinvR);

        // Set p = MinvR
        p = MinvR;
    };

    void Update_x_r(){
        float alpha = scalars[0]/scalars[2];
        for (unsigned i = 0; i < x_0.size(); ++i) {
            x_0[i] +=  alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
    };

    void Update_P(){
        float beta = scalars[1]/scalars[0];
        for (unsigned i = 0; i < p.size() ; ++i) {
            p[i] = MinvR[i] + beta * p[i];
        }
    };

    void print_banner(int i, array scalars) {
        std::cout << "Iteration " << i << ": " << "RTR = " <<  scalars[0]
                  << ", RTR_new = " << scalars[1]
                  << ", pTAp = " << scalars[2]
                  << ", alpha = " << scalars[0]/scalars[2]
                  << ", beta = " << scalars[1]/scalars[2]
                  << std::endl;
    }


    void solve_cg() {
        if (d[0]==0) {
            x = rhs;
            return;
        }

        PE(advance_integrate_matrix_solve_setup);
        Set_Up_CG();
        PL();

        for (int i = 1; i < 42; ++i){
            // Calculate AP
            PE(advance_integrate_matrix_solve_MV);
            MatrixVectorProduct(p, Ap, false);
            PL();

            // Calculate P^T*AP and store
            PE(advance_integrate_matrix_solve_VV);
            scalars[2] = std::inner_product(p.begin(), p.end(), Ap.begin(), value_type(0));

            // Calculate R^T*MinvR and store
            scalars[0] = std::inner_product(r.begin(), r.end(), MinvR.begin(), value_type(0));
            PL();

            // Update x and r
            PE(advance_integrate_matrix_solve_update);
            Update_x_r();
            PL();

            // Calculate new MinvR
            PE(advance_integrate_matrix_solve_tdma);
            solve(r, MinvR);
            PL();

            // Calculate new residual and store
            PE(advance_integrate_matrix_solve_VV);
            scalars[1] = std::inner_product(r.begin(), r.end(), MinvR.begin(), value_type(0));
            PL();

            if(sqrt(std::abs(scalars[1])) < 1e-11) {
                break;
            }

            // Update P and compute beta
            PE(advance_integrate_matrix_solve_update);
            Update_P();
            PL();
        }
        x = x_0;

        return;
    };

private:

    std::size_t size() const {
        return parent_index.size();
    }
};

} // namespace multicore
} // namespace arb
