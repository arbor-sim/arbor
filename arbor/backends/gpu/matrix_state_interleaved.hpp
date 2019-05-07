#pragma once

#include <arbor/assert.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/math.hpp>

#include "memory/memory.hpp"
#include "util/span.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/indirect.hpp"

#include "cuda_common.hpp"
#include "matrix_common.hpp"

namespace arb {
namespace gpu {

// host side wrapper for interleaved matrix assembly kernel
void assemble_matrix_interleaved(
    fvm_value_type* d,
    fvm_value_type* rhs,
    const fvm_value_type* invariant_d,
    const fvm_value_type* voltage,
    const fvm_value_type* current,
    const fvm_value_type* conductivity,
    const fvm_value_type* cv_capacitance,
    const fvm_value_type* area,
    const fvm_index_type* sizes,
    const fvm_index_type* starts,
    const fvm_index_type* matrix_to_cell,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* cell_to_intdom,
    unsigned padded_size, unsigned num_mtx);

// host side wrapper for interleaved matrix solver kernel
void solve_matrix_interleaved(
    fvm_value_type* rhs,
    fvm_value_type* d,
    const fvm_value_type* u,
    const fvm_index_type* p,
    const fvm_index_type* sizes,
    int padded_size,
    int num_mtx);

// host side wrapper for the flat to interleaved operation
void flat_to_interleaved(
    const fvm_value_type* in,
    fvm_value_type* out,
    const fvm_index_type* sizes,
    const fvm_index_type* starts,
    unsigned padded_size,
    unsigned num_vec);

// host side wrapper for the interleave to flat operation
void interleaved_to_flat(
    const fvm_value_type* in,
    fvm_value_type* out,
    const fvm_index_type* sizes,
    const fvm_index_type* starts,
    unsigned padded_size,
    unsigned num_vec);

// A helper that performs the interleave operation on host memory.
template <typename T, typename I>
std::vector<T> flat_to_interleaved(
        const std::vector<T>& in,
        const std::vector<I>& sizes,
        const std::vector<I>& starts,
        unsigned block_width, unsigned num_vec, unsigned padded_length)
{
    auto num_blocks = impl::block_count(num_vec, block_width);
    std::vector<T> out(num_blocks*block_width*padded_length, impl::npos<T>());
    for (auto mtx: util::make_span(0u, num_vec)) {
        auto block = mtx/block_width;
        auto lane  = mtx%block_width;

        auto len = sizes[mtx];
        auto src = starts[mtx];
        auto dst = block*(block_width*padded_length) + lane;
        for (auto i: util::make_span(0, len)) {
            out[dst] = in[src+i];
            dst += block_width;
        }
    }
    return out;
};

/// matrix state
template <typename T, typename I>
struct matrix_state_interleaved {
    using value_type = T;
    using index_type = I;

    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<index_type>;

    using const_view = typename array::const_view_type;

    // Permutation and index information required for forward and backward
    // interleave-permutation of vectors.

    // size of each matrix (after permutation in ascending size)
    iarray matrix_sizes;
    // start values corresponding to matrix i in the external storage
    iarray matrix_index;
    // map from matrix index (after permutation) to original cell index
    iarray matrix_to_cell_index;

    // Storage for the matrix and parent index in interleaved format.
    // Includes the cv_capacitance, which is required for matrix assembly.

    iarray parent_index;
    array d;   // [μS]
    array u;   // [μS]
    array rhs; // [nA]

    // required for matrix assembly
    array cv_capacitance; // [pF]

    // required for matrix assembly
    array cv_area; // [μm^2]

    // the invariant part of the matrix diagonal
    array invariant_d;    // [μS]

    iarray cell_to_intdom;

    // the length of a vector required to store values for one
    // matrix with padding
    unsigned padded_size;

    //  Storage for solution in uninterleaved format.
    //  Used to hold the storage for passing to caller, and must be updated
    //  after each call to the ::solve() method.
    array solution_;

    // default constructor
    matrix_state_interleaved() = default;

    // Construct matrix state for a set of matrices defined by parent_index p
    // The matrix solver stores the matrix in an "interleaved" structure for
    // optimal solution, which requires a significant amount of precomputing
    // of indexes and data structures in the constructor.
    //  cv_cap      // [pF]
    //  face_cond   // [μS]
    matrix_state_interleaved(const std::vector<index_type>& p,
                 const std::vector<index_type>& cell_cv_divs,
                 const std::vector<value_type>& cv_cap,
                 const std::vector<value_type>& face_cond,
                 const std::vector<value_type>& area,
                 const std::vector<index_type>& cell_intdom)
    {
        arb_assert(cv_cap.size()    == p.size());
        arb_assert(face_cond.size() == p.size());
        arb_assert(cell_cv_divs.back()  == (index_type)p.size());

        // Just because you never know.
        arb_assert(cell_cv_divs.size() <= UINT_MAX);

        using util::make_span;
        using util::indirect_view;

        // Convenience for commonly used type in this routine.
        using svec = std::vector<index_type>;

        //
        // Sort matrices in descending order of size.
        //

        // Find the size of each matrix.
        svec sizes;
        for (auto cv_span: util::partition_view(cell_cv_divs)) {
            sizes.push_back(cv_span.second-cv_span.first);
        }
        const auto num_mtx = sizes.size();

        // Find permutations and sort indexes/sizes.
        svec perm(num_mtx);
        std::iota(perm.begin(), perm.end(), 0);
        // calculate the permutation of matrices to put the in ascending size
        util::stable_sort_by(perm, [&sizes](index_type i){ return sizes[i]; });
        std::reverse(perm.begin(), perm.end());

        svec sizes_p = util::assign_from(indirect_view(sizes, perm));

        auto cell_to_cv = util::subrange_view(cell_cv_divs, 0, num_mtx);
        svec cell_to_cv_p = util::assign_from(indirect_view(cell_to_cv, perm));

        //
        // Calculate dimensions required to store matrices.
        //
        constexpr unsigned block_dim = impl::matrices_per_block();
        using impl::matrix_padding;

        // To start, take simplest approach of assuming all matrices stored
        // in blocks of the same dimension: padded_size
        padded_size = math::round_up(sizes_p[0], matrix_padding());
        const auto num_blocks = impl::block_count(num_mtx, block_dim);

        const auto total_storage = num_blocks*block_dim*padded_size;

        // calculate the interleaved and permuted p vector
        constexpr auto npos = std::numeric_limits<index_type>::max();
        std::vector<index_type> p_tmp(total_storage, npos);
        for (auto mtx: make_span(0, num_mtx)) {
            auto block = mtx/block_dim;
            auto lane  = mtx%block_dim;

            auto len = sizes_p[mtx];
            auto src = cell_to_cv_p[mtx];
            auto dst = block*(block_dim*padded_size) + lane;
            for (auto i: make_span(0, len)) {
                // the p indexes are always relative to the start of the p vector.
                // the addition and subtraction of dst and src respectively is to convert from
                // the original offset to the new padded and permuted offset.
                p_tmp[dst+block_dim*i] = dst + block_dim*(p[src+i]-src);
            }
        }

        d   = array(total_storage);
        u   = array(total_storage);
        rhs = array(total_storage);
        parent_index = memory::make_const_view(p_tmp);

        //
        //  Calculate the invariant part of the matrix diagonal and the
        //  upper diagonal on the host, then copy to the device.
        //

        std::vector<value_type> invariant_d_tmp(p.size(), 0);
        std::vector<value_type> u_tmp(p.size(), 0);
        auto face_conductance_tmp = memory::on_host(face_cond);
        for (auto i: util::make_span(1u, p.size())) {
            auto gij = face_conductance_tmp[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }

        // Helper that converts to interleaved format on the host, then copies to device
        // memory, for use as an rvalue in an assignemt to a device vector.
        auto interleave = [&] (std::vector<T>const& x) {
            return memory::on_gpu(
                flat_to_interleaved(x, sizes_p, cell_to_cv_p, block_dim, num_mtx, padded_size));
        };
        u              = interleave(u_tmp);
        invariant_d    = interleave(invariant_d_tmp);
        cv_area        = interleave(area);
        cv_capacitance = interleave(cv_cap);

        matrix_sizes = memory::make_const_view(sizes_p);
        matrix_index = memory::make_const_view(cell_to_cv_p);
        matrix_to_cell_index = memory::make_const_view(perm);
        cell_to_intdom = memory::make_const_view(cell_intdom);

        // Allocate space for storing the un-interleaved solution.
        solution_ = array(p.size());
    }

    const_view solution() const {
        return solution_;
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current.
    //   dt_intdom         [ms]     (per integration domain)
    //   voltage           [mV]     (per compartment)
    //   current density   [A/m²]   (per compartment)
    //   conductivity      [kS/m²]  (per compartment)
    void assemble(const_view dt_intdom, const_view voltage, const_view current, const_view conductivity) {
        assemble_matrix_interleaved
            (d.data(), rhs.data(), invariant_d.data(),
             voltage.data(), current.data(), conductivity.data(), cv_capacitance.data(), cv_area.data(),
             matrix_sizes.data(), matrix_index.data(),
             matrix_to_cell_index.data(),
             dt_intdom.data(), cell_to_intdom.data(), padded_matrix_size(), num_matrices());

    }

    void solve() {
        // Perform the Hines solve.
        solve_matrix_interleaved(
             rhs.data(), d.data(), u.data(), parent_index.data(), matrix_sizes.data(),
             padded_matrix_size(), num_matrices());

        // Copy the solution from interleaved to front end storage.
        interleaved_to_flat
            (rhs.data(), solution_.data(), matrix_sizes.data(), matrix_index.data(),
             padded_matrix_size(), num_matrices());
    }

private:

    // The number of matrices stored in the matrix state.
    unsigned num_matrices() const {
        return matrix_sizes.size();
    }

    // The full padded matrix size
    unsigned padded_matrix_size() const {
        return padded_size;
    }
};

} // namespace gpu
} // namespace arb
