#pragma once

#include <arbor/fvm_types.hpp>

#include "memory/memory.hpp"
#include "memory/wrappers.hpp"
#include "util/span.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace gpu {

// CUDA implementation entry points:

void solve_matrix_flat(
    fvm_value_type* rhs,
    fvm_value_type* d,
    const fvm_value_type* u,
    const fvm_index_type* p,
    const fvm_index_type* cell_cv_divs,
    int num_mtx);

void assemble_matrix_flat(
    fvm_value_type* d,
    fvm_value_type* rhs,
    const fvm_value_type* invariant_d,
    const fvm_value_type* voltage,
    const fvm_value_type* current,
    const fvm_value_type* conductivity,
    const fvm_value_type* cv_capacitance,
    const fvm_value_type* cv_area,
    const fvm_index_type* cv_to_cell,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* cell_to_intdom,
    unsigned n);

/// matrix state
template <typename T, typename I>
struct matrix_state_flat {
    using value_type = T;
    using index_type = I;

    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<index_type>;

    using view = typename array::view_type;
    using const_view = typename array::const_view_type;

    iarray parent_index;
    iarray cell_cv_divs;
    iarray cv_to_cell;

    array d;     // [μS]
    array u;     // [μS]
    array rhs;   // [nA]

    array cv_capacitance;    // [pF]
    array face_conductance;  // [μS]
    array cv_area;           // [μm^2]

    iarray cell_to_intdom;

    // the invariant part of the matrix diagonal
    array invariant_d;         // [μS]

    matrix_state_flat() = default;

    matrix_state_flat(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cv_cap,
                 const std::vector<value_type>& face_cond,
                 const std::vector<value_type>& area,
                 const std::vector<index_type>& cell_to_intdom):
        parent_index(memory::make_const_view(p)),
        cell_cv_divs(memory::make_const_view(cell_cv_divs)),
        cv_to_cell(p.size()),
        d(p.size()),
        u(p.size()),
        rhs(p.size()),
        cv_capacitance(memory::make_const_view(cv_cap)),
        cv_area(memory::make_const_view(area)),
        cell_to_intdom(memory::make_const_view(cell_to_intdom))
    {
        arb_assert(cv_cap.size() == size());
        arb_assert(face_cond.size() == size());
        arb_assert(area.size() == size());
        arb_assert(cell_cv_divs.back() == (index_type)size());
        arb_assert(cell_cv_divs.size() > 1u);

        using memory::make_const_view;

        auto n = d.size();
        std::vector<index_type> cv_to_cell_tmp(n, 0);
        std::vector<value_type> invariant_d_tmp(n, 0);
        std::vector<value_type> u_tmp(n, 0);

        for (auto i: util::make_span(1u, n)) {
            auto gij = face_cond[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }

        index_type ci = 0;
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            util::fill(util::subrange_view(cv_to_cell_tmp, cv_span), ci);
            ++ci;
        }

        cv_to_cell = make_const_view(cv_to_cell_tmp);
        invariant_d = make_const_view(invariant_d_tmp);
        u = make_const_view(u_tmp);
    }

    // interface for exposing the solution to the outside world
    const_view solution() const {
        return memory::make_view(rhs);
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
    //   dt_intdom [ms] (per integration domain)
    //   voltage   [mV]
    //   current   [nA]
    void assemble(const_view dt_intdom, const_view voltage, const_view current, const_view conductance) {
        // perform assembly on the gpu
        assemble_matrix_flat(
            d.data(), rhs.data(), invariant_d.data(), voltage.data(),
            current.data(), conductance.data(), cv_capacitance.data(), cv_area.data(),
            cv_to_cell.data(), dt_intdom.data(), cell_to_intdom.data(), size());
    }

    void solve() {
        // perform solve on gpu
        solve_matrix_flat(rhs.data(), d.data(), u.data(), parent_index.data(),
                          cell_cv_divs.data(), num_matrices());
    }

    std::size_t size() const {
        return parent_index.size();
    }

private:
    unsigned num_matrices() const {
        return cell_cv_divs.size()-1;
    }
};

} // namespace gpu
} // namespace arb
