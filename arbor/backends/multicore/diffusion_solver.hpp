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

    array d;           // [μS]
    array u;           // [μS]
    array cv_volume;   // [μm^3]
    array invariant_d; // [μS] invariant part of matrix diagonal

    diffusion_solver() = default;
    diffusion_solver(const diffusion_solver&) = default;
    diffusion_solver(diffusion_solver&&) = default;

    diffusion_solver& operator=(const diffusion_solver&) = default;
    diffusion_solver& operator=(diffusion_solver&&) = default;

    diffusion_solver(const std::vector<index_type>& p,
                     const std::vector<index_type>& cell_cv_divs,
                     const std::vector<value_type>& diff,
                     const std::vector<value_type>& volume):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        d(size(), 0), u(size(), 0),
        cv_volume(volume.begin(), volume.end()),
        invariant_d(size(), 0) {
        // Sanity check
        arb_assert(diff.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());

        // Build invariant parts
        const auto n = size();
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


    // Assemble and solve the matrix
    // Assemble the matrix
    //   concentration   [mM]      (per control volume)
    //   dt              [ms]      (scalar)
    template<typename T>
    void solve(T& concentration,
               value_type dt) {
        auto cell_cv_part = util::partition_view(cell_cv_divs);
        index_type ncells = cell_cv_part.size();

        value_type _1_dt = 1e-3/dt;         // 1/µs
        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            for (auto i: util::make_span(cell_cv_part[m])) {
                auto X = concentration[i];  // mM
                auto V = cv_volume[i];      // m^3
                d[i]   = _1_dt*V + invariant_d[i];
                concentration[i] = _1_dt*V*X;
            }
        }
        solve(concentration);
    }

    // Separate solver; analoguos with cable solver
    template<typename T>
    void solve(T& rhs) {
        // loop over submatrices
        for (const auto& [first, last]: util::partition_view(cell_cv_divs)) {
            if (first >= last || d[first] == 0) continue; // skip cell with no CVs

            // backward sweep
            for(int i = last - 1; i >= first; --i) {
                const auto pi = parent_index[i];
                const auto factor = u[i] / d[i];
                d[pi] -= factor * u[i];
                rhs[pi] -= factor * rhs[i];
            }

            // solve root
            rhs[first] /= d[first];

            // forward sweep
            for(int i = first + 1; i < last; ++i) {
                auto pi = parent_index[i];
                rhs[i] = (rhs[i] - u[i] * rhs[pi])/d[i];
            }
        }
    }

    std::size_t size() const { return parent_index.size(); }
};

} // namespace multicore
} // namespace arb
