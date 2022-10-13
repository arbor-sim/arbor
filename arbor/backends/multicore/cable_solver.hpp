#pragma once

#include <util/partition.hpp>
#include <util/span.hpp>

#include <memory/memory.hpp>

#include "multicore_common.hpp"

namespace arb {
namespace multicore {

struct cable_solver {
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using array      = padded_vector<value_type>;
    using const_view = const array&;
    using iarray     = padded_vector<index_type>;

    iarray parent_index;
    iarray cell_cv_divs;
    iarray cell_to_intdom;

    array d;              // [μS]
    array u;              // [μS]
    array cv_capacitance; // [pF]
    array cv_area;        // [μm^2]
    array invariant_d;    // [μS] invariant part of matrix diagonal

    cable_solver() = default;
    cable_solver(const cable_solver&) = default;
    cable_solver(cable_solver&&) = default;

    cable_solver& operator=(const cable_solver&) = default;
    cable_solver& operator=(cable_solver&&) = default;

    cable_solver(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond,
                 const std::vector<value_type>& area,
                 const std::vector<index_type>& cell_to_intdom):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        cell_to_intdom(cell_to_intdom.begin(), cell_to_intdom.end()),
        d(size(), 0), u(size(), 0),
        cv_capacitance(cap.begin(), cap.end()),
        cv_area(area.begin(), area.end()),
        invariant_d(size(), 0)
    {
        // Sanity check
        arb_assert(cap.size() == size());
        arb_assert(cond.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());

        // Build invariant parts
        if (size() >= 1) {
            for (auto i: util::make_span(1u, size())) {
                const auto gij = cond[i];
                u[i] = -gij;
                invariant_d[i] += gij;
                if (p[i]!=-1) { // root
                    invariant_d[p[i]] += gij;
                }
            }
        }
    }

    // Setup and solve the cable equation
    // * expects the voltage from its first argument
    // * will likewise overwrite the first argument with the solction
    template<typename T>
    void solve(T& rhs, const_view dt_intdom, const_view current, const_view conductivity) {
        value_type * const ARB_NO_ALIAS d_ = d.data();
        value_type * const ARB_NO_ALIAS r_ = rhs.data();

        const value_type * const ARB_NO_ALIAS i_ = current.data();
        const value_type * const ARB_NO_ALIAS inv_ = invariant_d.data();
        const value_type * const ARB_NO_ALIAS c_ = cv_capacitance.data();
        const value_type * const ARB_NO_ALIAS g_ = conductivity.data();
        const value_type * const ARB_NO_ALIAS a_ = cv_area.data();

        const auto cell_cv_part = util::partition_view(cell_cv_divs);
        const index_type ncells = cell_cv_part.size();
        // Assemble; loop over submatrices
        // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
        //   dt_intdom       [ms]      (per integration domain)
        //   voltage         [mV]      (per control volume)
        //   current density [A.m^-2]  (per control volume)
        //   conductivity    [kS.m^-2] (per control volume)
        for (auto m: util::make_span(0, ncells)) {
            const auto dt = dt_intdom[cell_to_intdom[m]];    // [ms]
            if (dt > 0) {
                const value_type oodt = 1e-3/dt;             // [1/µs]
                const auto& [lo, hi] = cell_cv_part[m];
                for(int i = lo; i < hi; ++i) {
                    const auto area = 1e-3*a_[i];            // [1e-9·m²]
                    const auto gi = oodt*c_[i] + area*g_[i]; // [μS]
                    d_[i] = gi + inv_[i];                    // [μS]
                    r_[i] = gi*r_[i] - area*i_[i];           // [nA]
                }
            }
            else {
                const auto& [lo, hi] = cell_cv_part[m];
                for(int i = lo; i < hi; ++i) {
                    d_[i] = 0.0;
                }
            }
        }
        solve(rhs);
    }

    // Solve; loop over submatrices
    // Afterwards rhs will contain the solution.
    // NOTE: This exists separately only to cater to the tests
    template<typename T>
    void solve(T& rhs) {
        value_type * const ARB_NO_ALIAS r_ = rhs.data();
        value_type * const ARB_NO_ALIAS d_ = d.data();

        const value_type * const ARB_NO_ALIAS u_ = u.data();
        const index_type * const ARB_NO_ALIAS p_ = parent_index.data();

        const auto cell_cv_part = util::partition_view(cell_cv_divs);
        for (const auto& [first, last]: cell_cv_part) {
            if (first < last && d_[first] != 0) {  // skip vacuous cells
                // backward sweep
                for(int i = last - 1; i > first; --i) {
                    const auto factor = u_[i] / d_[i];
                    const auto pi = p_[i];
                    d_[pi] -= factor * u_[i];
                    r_[pi] -= factor * r_[i];
                }
                // solve root
                r_[first] /= d_[first];
                // forward sweep
                for(int i = first + 1; i < last; ++i) {
                    r_[i] -= u_[i] * r_[p_[i]];
                    r_[i] /= d_[i];
                }
            }
        }
    }

    std::size_t num_cells() const { return cell_cv_divs.size() - 1; }
    std::size_t size() const { return parent_index.size(); }
};

} // namespace multicore
} // namespace arb
