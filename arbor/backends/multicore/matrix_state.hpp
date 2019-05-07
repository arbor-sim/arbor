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

    array cv_capacitance;      // [pF]
    array face_conductance;    // [μS]
    array cv_area;             // [μm^2]

    iarray cell_to_intdom;

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    matrix_state() = default;

    matrix_state(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond,
                 const std::vector<value_type>& area,
                 const std::vector<index_type>& cell_to_intdom):
        parent_index(p.begin(), p.end()),
        cell_cv_divs(cell_cv_divs.begin(), cell_cv_divs.end()),
        d(size(), 0), u(size(), 0), rhs(size()),
        cv_capacitance(cap.begin(), cap.end()),
        face_conductance(cond.begin(), cond.end()),
        cv_area(area.begin(), area.end()),
        cell_to_intdom(cell_to_intdom.begin(), cell_to_intdom.end())
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
        return rhs;
    }


    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
    //   dt_intdom       [ms]      (per integration domain)
    //   voltage         [mV]      (per control volume)
    //   current density [A.m^-2]  (per control volume)
    //   conductivity    [kS.m^-2] (per control volume)
    void assemble(const_view dt_intdom, const_view voltage, const_view current, const_view conductivity) {
        auto cell_cv_part = util::partition_view(cell_cv_divs);
        const index_type ncells = cell_cv_part.size();

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto dt = dt_intdom[cell_to_intdom[m]];

            if (dt>0) {
                value_type oodt_factor = 1e-3/dt; // [1/µs]
                for (auto i: util::make_span(cell_cv_part[m])) {
                    auto area_factor = 1e-3*cv_area[i]; // [1e-9·m²]

                    auto gi = oodt_factor*cv_capacitance[i] + area_factor*conductivity[i]; // [μS]

                    d[i] = gi + invariant_d[i];
                    // convert current to units nA
                    rhs[i] = gi*voltage[i] - area_factor*current[i];
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

    void solve() {
        // loop over submatrices
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            auto first = cv_span.first;
            auto last = cv_span.second; // one past the end

            if (d[first]!=0) {
                // backward sweep
                for(auto i=last-1; i>first; --i) {
                    auto factor = u[i] / d[i];
                    d[parent_index[i]]   -= factor * u[i];
                    rhs[parent_index[i]] -= factor * rhs[i];
                }
                rhs[first] /= d[first];

                // forward sweep
                for(auto i=first+1; i<last; ++i) {
                    rhs[i] -= u[i] * rhs[parent_index[i]];
                    rhs[i] /= d[i];
                }
            }
        }
    }

private:

    std::size_t size() const {
        return parent_index.size();
    }
};

} // namespace multicore
} // namespace arb
