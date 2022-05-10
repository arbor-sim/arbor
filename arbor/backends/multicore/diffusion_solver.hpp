#pragma once

#include <util/partition.hpp>
#include <util/span.hpp>

#include <memory/memory.hpp>

#include "multicore_common.hpp"

namespace arb {
namespace multicore {

struct diffusion_solver {
    using value_type = arb_value_type;
    using index_type = arb_index_type;
    using array      = padded_vector<value_type>;
    using const_view = const array&;
    using iarray     = padded_vector<index_type>;

    iarray parent_index;
    iarray cell_cv_divs;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    array cv_area;             // [μm^2]

    iarray cell_to_intdom;

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    diffusion_solver() = default;
    diffusion_solver(const diffusion_solver&) = default;
    diffusion_solver(diffusion_solver&&) = default;

    diffusion_solver& operator=(const diffusion_solver&) = default;
    diffusion_solver& operator=(diffusion_solver&&) = default;

    diffusion_solver(const std::vector<index_type>& p,
                     const std::vector<index_type>& cell_cv_divs,
                     const std::vector<value_type>& diff,
                     const std::vector<value_type>& area,
                     const std::vector<index_type>& cell_to_intdom):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        d(size(), 0), u(size(), 0), rhs(size()),
        cv_area(area.begin(), area.end()),
        cell_to_intdom(cell_to_intdom.begin(), cell_to_intdom.end())
    {
        // Sanity check
        arb_assert(diff.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());

        // Build invariant parts
        const auto n = size();
        invariant_d = array(n, 0);
        if (n >= 1) { // skip empty matrix, ie cell with empty morphology
            for (auto i: util::make_span(1u, n)) {
                auto gij = diff[i];
                u[i]           =  -gij;
                invariant_d[i] +=  gij;
                // Also add to our parent, if present
                if (auto pi = p[i]; pi != -1) invariant_d[pi] += gij;
            }
        }
    }

    const_view solution() const {
        // In this back end the solution is a simple view of the rhs, which
        // contains the solution after the matrix_solve is performed.
        return rhs;
    }

    // Assemble the matrix
    //   dt_intdom       [ms]      (per integration domain)
    //   internal conc   [mM]      (per control volume)
    //   voltage         [mV]      (per control volume)
    //   current density [A.m^-2]  (per control volume and species)
    //   diffusivity     [???]     (per control volume)
    //   charge          [e]
    void assemble(const_view dt_intdom, const_view concentration, const_view voltage, const_view current, const_view conductivity, fvm_value_type q) {
        auto cell_cv_part = util::partition_view(cell_cv_divs);
        index_type ncells = cell_cv_part.size();
        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto dt = dt_intdom[cell_to_intdom[m]];
            if (dt>0) {
                value_type _1_dt = 1e-3/dt;     // 1/µs
                for (auto i: util::make_span(cell_cv_part[m])) {
                    auto u = voltage[i];        // mV
                    auto g = conductivity[i];   // µS
                    auto J = current[i];        // A/m^2
                    auto A = 1e-3*cv_area[i];   // 1e-9·m²
                    auto X = concentration[i];  // mM
                    // conversion from current density to concentration change
                    // using Faraday's constant
                    auto F = A/(q*96.485332);
                    d[i]   = _1_dt   + F*g + invariant_d[i];
                    rhs[i] = _1_dt*X + F*(u*g - J);
                }
            }
            else {
                for (auto i: util::make_span(cell_cv_part[m])) {
                    d[i]   = 0;
                    rhs[i] = concentration[i];
                }
            }
        }
    }

    void solve() {
        // loop over submatrices
        for (const auto& [first, last]: util::partition_view(cell_cv_divs)) {
            if (first >= last) continue; // skip cell with no CVs
            if (d[first]!=0) {
                // backward sweep
                for(auto i=last-1; i>first; --i) {
                    auto pi  = parent_index[i];
                    auto fac = -u[i] / d[i];
                    d[pi]    = std::fma(fac, u[i],   d[pi]);
                    rhs[pi]  = std::fma(fac, rhs[i], rhs[pi]);
                }
                // solve root
                rhs[first] /= d[first];
                // forward sweep
                for(auto i=first+1; i<last; ++i) {
                    rhs[i] = std::fma(-u[i], rhs[parent_index[i]], rhs[i])/d[i];
                }
            }
        }
    }

    template<typename VTo>
    void solve(VTo& to) {
        solve();
        memory::copy(rhs, to);
    }

    std::size_t size() const { return parent_index.size(); }
};

} // namespace multicore
} // namespace arb
