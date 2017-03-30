#pragma once

#include <memory/memory.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "gpu_kernels/solve_matrix.hpp"
#include "gpu_kernels/assemble_matrix.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// matrix state
template <typename T, typename I>
struct matrix_state_flat {
    using value_type = T;
    using size_type = I;

    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view = typename array::view_type;
    using const_view = typename array::const_view_type;

    iarray parent_index;
    iarray cell_index;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    array cv_capacitance;      // [pF]
    array face_conductance;    // [μS]

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    // interface for exposing the solution to the outside world
    view solution;

    matrix_state_flat() = default;

    matrix_state_flat(const std::vector<size_type>& p,
                 const std::vector<size_type>& cell_idx,
                 const std::vector<value_type>& cv_cap,
                 const std::vector<value_type>& face_cond):
        parent_index(memory::make_const_view(p)),
        cell_index(memory::make_const_view(cell_idx)),
        d(p.size()),
        u(p.size()),
        rhs(p.size()),
        cv_capacitance(memory::make_const_view(cv_cap))
    {
        EXPECTS(cv_cap.size() == size());
        EXPECTS(face_cond.size() == size());
        EXPECTS(cell_idx.back() == size());

        using memory::make_const_view;

        auto n = d.size();
        std::vector<value_type> invariant_d_tmp(n, 0);
        std::vector<value_type> u_tmp(n, 0);

        for(auto i: util::make_span(1u, n)) {
            auto gij = face_cond[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }
        invariant_d = make_const_view(invariant_d_tmp);
        u = make_const_view(u_tmp);

        solution = rhs;
    }

    int num_matrices() const {
        return cell_index.size()-1;
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current
    //   dt      [ms]
    //   voltage [mV]
    //   current [nA]
    void assemble(value_type dt, const_view voltage, const_view current) {
        // determine the grid dimensions for the kernel
        auto const n = voltage.size();
        auto const block_dim = 128;
        auto const grid_dim = impl::block_count(n, block_dim);

        assemble_matrix_flat<value_type, size_type><<<grid_dim, block_dim>>> (
            d.data(), rhs.data(), invariant_d.data(), voltage.data(),
            current.data(), cv_capacitance.data(), dt, size());
    }

    void solve() {
        // determine the grid dimensions for the kernel
        auto const block_dim = 128;
        auto const grid_dim = impl::block_count(num_matrices(), block_dim);

        // perform solve on gpu
        solve_matrix_flat<value_type, size_type><<<grid_dim, block_dim>>> (
            rhs.data(), d.data(), u.data(), parent_index.data(),
            cell_index.data(), num_matrices());
    }

    std::size_t size() const {
        return parent_index.size();
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
