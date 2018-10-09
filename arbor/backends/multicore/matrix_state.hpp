#pragma once

#include <util/partition.hpp>
#include <util/span.hpp>

#include "multicore_common.hpp"

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

    matrix_state() = default;

    matrix_state(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond,
                 const std::vector<value_type>& area):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        d(size(), 0), u(size(), 0), rhs(size()), x(size()),
        cv_capacitance(cap.begin(), cap.end()),
        face_conductance(cond.begin(), cond.end()),
        cv_area(area.begin(), area.end())
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
        x = b;
    }

    void solve() {
        x = rhs;
        for(unsigned i = 0; i < size(); i++) {
            if(x[i]!=rhs[i]) {
                std::cout<< "err at "<< i<< std::endl;
                break;
            }
        }
        solve_tdma();
    }

    void MatrixVectorProduct(const array& b, array& c){
        for (unsigned i = 0; i < size(); i++) {
            c[i] = d[i] * b[i];
        }
        for (unsigned i = 1; i < size(); i++) {
            unsigned p = parent_index[i];
            c[p] += u[i] * b[i];
            c[i] += u[i] * b[p];
        }
        /*for (auto e: A.e) {
            c[e.coords.first] += e.weight * b[e.coords.second];
        }*/
    };

    void Set_Up_CG(const array& b, array& r, array& p, array& MinvR){

        // Solve Mx = b; Solution is Minvb = x(0)
        array Minvb(size());
        solve(b, Minvb);

        // R(0) = b - A*x(0) = b - A*MinvB
        MatrixVectorProduct(Minvb, r);

        // Solve Mx = R; store in MinvR
        solve(r, MinvR);

        // P(0) = MinvR
        p = MinvR;
    };

    void DotProduct(const array& a, const array& b, array& scalars, unsigned pos){
        value_type sum = 0.;
        for (unsigned i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        scalars[pos] = sum;
    };

    void Update_x(array& x, const array& p, array& scalars, unsigned pos){
        float alpha = scalars[0]/scalars[2];
        scalars[pos] = alpha;

        for (unsigned i = 0; i < x.size(); ++i) {
            x[i] +=  alpha * p[i];
        }
    };

    void Update_R(array& r, const array& Ap, const array& scalars){
        float alpha = scalars[3];

        for (unsigned i = 0; i < r.size(); ++i) {
            r[i] -= alpha*Ap[i];
        }
    };

    void Update_P(array& p, const array& r, array& scalars, unsigned pos){

        float beta = scalars[1]/scalars[0];

        scalars[pos] = beta;

        for (unsigned i = 0; i < p.size() ; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    };

    void print_banner(int i, array scalars) {
        std::cout << "Iteration " << i << ": " << "RTR = " <<  scalars[0]
                  << ", RTR_new = " << scalars[1]
                  << ", pTAp = " << scalars[2]
                  << ", alpha = " << scalars[3]
                  << ", beta = " << scalars[4]
                  << std::endl;
    }


    void solve_cg() {
        array r(size()), p(size()), Ap(size()), MinvR(size());
        array scalars(5);	// array of variables (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

        Set_Up_CG(rhs, r, p, MinvR);

        for (int i = 1; i < 42; ++i){

            // Calculate AP
            MatrixVectorProduct(p, Ap);

            // Calculate P^T*AP and store
            DotProduct(p, Ap, scalars, 2);     // elements placed in scalars[2] (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

            // Calculate R^T*MinvR and store
            DotProduct(r, MinvR, scalars, 0);    // elements placed in scalars[0] (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

            // Update x and store alpha
            Update_x(x, p, scalars, 3);          // elements placed in scalars[3] (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

            // Update R
            Update_R(r, Ap, scalars);

            // Calculate new MinvR
            solve(r, MinvR);

            // Calculate new residual and store
            DotProduct(r, MinvR, scalars, 1);    // elements placed in scalars[1] (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)

            // Check residual threshold and check scalars
            print_banner(i, scalars);

            if(sqrt(std::abs(scalars[1])) < 1e-11) {
                break;
            }

            // Update P and compute beta
            Update_P(p, MinvR, scalars, 4);          // elements placed in scalars[4] (RTMinvR, RTMinvR_new, P^TAP, alpha, beta)
        }

        solve_tdma();
        std::cout<<std::endl;
        std::cout<<std::endl;

        for(unsigned i = 0; i< x.size(); i++) {
            std::cout << x[i] << " ";
        }

        std::cout<<std::endl;
        std::cout<<std::endl;

        return;
    };

private:

    std::size_t size() const {
        return parent_index.size();
    }
};

} // namespace multicore
} // namespace arb
