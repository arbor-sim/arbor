#pragma once

#include <memory/memory.hpp>
#include <util/partition.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {
namespace multicore {

template <typename T, typename I>
struct matrix_state {
public:
    using value_type = T;
    using size_type = I;

    using array = memory::host_vector<value_type>;
    using const_view = typename array::const_view_type;
    using iarray = memory::host_vector<size_type>;
    iarray parent_index;
    iarray cell_index;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    array cv_capacitance;      // [pF]
    array face_conductance;    // [μS]

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    const_view solution;

    matrix_state() = default;

    matrix_state(const std::vector<size_type>& p,
                 const std::vector<size_type>& cell_idx,
                 const std::vector<value_type>& cap,
                 const std::vector<value_type>& cond):
        parent_index(memory::make_const_view(p)),
        cell_index(memory::make_const_view(cell_idx)),
        d(size(), 0), u(size(), 0), rhs(size()),
        cv_capacitance(memory::make_const_view(cap)),
        face_conductance(memory::make_const_view(cond))
    {
        EXPECTS(cap.size() == size());
        EXPECTS(cond.size() == size());
        EXPECTS(cell_idx.back() == size());

        auto n = size();
        invariant_d = array(n, 0);
        for (auto i: util::make_span(1u, n)) {
            auto gij = face_conductance[i];

            u[i] = -gij;
            invariant_d[i] += gij;
            invariant_d[p[i]] += gij;
        }

        // In this back end the solution is a simple view of the rhs, which
        // contains the solution after the matrix_solve is performed.
        solution = rhs;
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current
    //   time    [ms]
    //   time_to [ms]
    //   voltage [mV]
    //   current [nA]
    void assemble(const_view time, const_view time_to, const_view voltage, const_view current) {
        auto cell_part = util::partition_view(cell_index);
        const size_type ncells = cell_part.size();

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto dt = time_to[m]-time[m];
            value_type factor = 1e-3/dt;

            for (auto i: util::make_span(cell_part[m])) {
                auto gi = factor*cv_capacitance[i];

                d[i] = gi + invariant_d[i];
                rhs[i] = gi*voltage[i] - current[i];
            }
        }
    }

    void solve() {
        const size_type ncells = cell_index.size()-1;

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto first = cell_index[m];
            auto last = cell_index[m+1];

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

private:

    std::size_t size() const {
        return parent_index.size();
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
