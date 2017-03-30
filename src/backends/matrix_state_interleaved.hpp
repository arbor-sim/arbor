#pragma once

#include <memory/memory.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "gpu_kernels.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// matrix state
template <typename T, typename I>
struct matrix_state_interleaved {
    using value_type = T;
    using size_type = I;

    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using const_view = typename array::const_view_type;

    //
    // Permutation and index information required for forward and backward
    // interleave-permutation of vectors.
    //

    // size of each matrix (after permutation in ascending size)
    iarray matrix_sizes;
    // start values corresponding to matrix i in the external storage
    iarray matrix_index;

    //
    // Storage for the matrix and parent index in interleaved format.
    // Includes the cv_capacitance, which is required for matrix assembly.
    //
    iarray parent_index;
    array d;   // [μS]
    array u;   // [μS]
    array rhs; // [nA]

    // required for matrix assembly
    array cv_capacitance; // [pF]

    // the invariant part of the matrix diagonal
    array invariant_d;    // [μS]

    // the length of a vector required to store values for one
    // matrix with padding
    int padded_size;

    //  Storage for solution in uninterleaved format.
    //  Used to hold the storage for passing to caller, and must be updated
    //  after each call to the ::solve() method.
    array solution;

    // default constructor
    matrix_state_interleaved() = default;

    // Construct matrix state for a set of matrices defined by parent_index p
    // The matrix solver stores the matrix in an "interleaved" structure for
    // optimal solution, which requires a significant amount of precomputing
    // of indexes and data structures in the constructor.
    //  cv_cap      // [pF]
    //  face_cond   // [μS]
    matrix_state_interleaved(const std::vector<size_type>& p,
                 const std::vector<size_type>& cell_idx,
                 const std::vector<value_type>& cv_cap,
                 const std::vector<value_type>& face_cond)
    {
        EXPECTS(cv_cap.size()    == p.size());
        EXPECTS(face_cond.size() == p.size());
        EXPECTS(cell_idx.back()  == p.size());

        using util::make_span;

        // convenience for commonly used type in this routine
        using svec = std::vector<size_type>;

        //
        // sort matrices in descending order of size
        //

        // find the size of each matrix
        const auto num_mtx = cell_idx.size()-1;
        svec sizes;
        for (auto it=cell_idx.begin()+1; it!=cell_idx.end(); ++it) {
            sizes.push_back(*it - *(it-1));
        }

        // find permutations and sort indexes/sizes
        svec perm(num_mtx);
        std::iota(perm.begin(), perm.end(), 0);
        // calculate the permutation of matrices to put the in ascending size
        util::stable_sort_by(perm, [&sizes](size_type i){ return sizes[i]; });
        std::reverse(perm.begin(), perm.end());

        // TODO: refactor to be less verbose with permutation_view
        svec sizes_p;
        for (auto i: make_span(0, num_mtx)) {
            sizes_p.push_back(sizes[perm[i]]);
        }
        svec cell_index_p;
        for (auto i: make_span(0, num_mtx)) {
            cell_index_p.push_back(cell_idx[perm[i]]);
        }

        //
        // Calculate dimensions required to store matrices.
        //
        using impl::block_dim;
        using impl::matrix_padding;

        // To start, take simplest approach of assuming all matrices stored
        // in blocks of the same dimension: padded_size
        padded_size = impl::padded_size(sizes_p[0], matrix_padding());
        const auto num_blocks = impl::block_count(num_mtx, block_dim());

        const auto total_storage = num_blocks*block_dim()*padded_size;

        // calculate the interleaved and permuted p vector
        constexpr auto npos = std::numeric_limits<size_type>::max();
        std::vector<size_type> p_tmp(total_storage, npos);
        for (auto mtx: make_span(0, num_mtx)) {
            auto block = mtx/block_dim();
            auto lane  = mtx%block_dim();

            auto len = sizes_p[mtx];
            auto src = cell_index_p[mtx];
            auto dst = block*(block_dim()*padded_size) + lane;
            for (auto i: make_span(0, len)) {
                // the p indexes are always relative to the start of the p vector.
                // the addition and subtraction of dst and src respectively is to convert from
                // the original offset to the new padded and permuted offset.
                p_tmp[dst+block_dim()*i] = dst + block_dim()*(p[src+i]-src);
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
        for(auto i: util::make_span(1u, p.size())) {
            auto gij = face_conductance_tmp[i];

            u_tmp[i] = -gij;
            invariant_d_tmp[i] += gij;
            invariant_d_tmp[p[i]] += gij;
        }

        // helper that converts to interleaved format on the host, then copies to device
        // memory, for use as an rvalue in an assignemt to a device vector.
        auto interleave = [&] (std::vector<T>const& x) {
            return memory::on_gpu(
                interleave_host(x, sizes_p, cell_index_p, block_dim(), num_mtx, padded_size));
        };
        u = interleave(u_tmp);
        invariant_d = interleave(invariant_d_tmp);
        cv_capacitance = interleave(cv_cap);

        matrix_sizes = memory::make_const_view(sizes_p);
        matrix_index = memory::make_const_view(cell_index_p);

        // allocate space for storing the un-interleaved solution
        solution = array(p.size());
    }

    // the number of matrices stored in the matrix state
    int num_matrices() const {
        return matrix_sizes.size();
    }

    // the full padded matrix size
    int padded_matrix_size() const {
        return padded_size;
    }

    // Assemble the matrix
    // Afterwards the diagonal and RHS will have been set given dt, voltage and current
    //   dt      [ms]
    //   voltage [mV]
    //   current [nA]
    void assemble(value_type dt, const_view voltage, const_view current) {
        constexpr auto bd = impl::block_dim();
        constexpr auto lw = impl::load_width();
        constexpr auto block_dim = bd*lw;

        // the number of threads is threads_per_matrix*num_mtx
        const auto num_blocks = impl::block_count(num_matrices()*lw, block_dim);

        assemble_matrix_interleaved<value_type, size_type, bd, lw, block_dim>
            <<<num_blocks, block_dim>>>
            ( d.data(), rhs.data(), invariant_d.data(),
              voltage.data(), current.data(), cv_capacitance.data(),
              matrix_sizes.data(), matrix_index.data(),
              dt, padded_matrix_size(), num_matrices());

    }

    void solve() {
        // perform the Hines solve
        auto const grid_dim = impl::block_count(num_matrices(), impl::block_dim());

        solve_matrix_interleaved<value_type, size_type, impl::block_dim()>
            <<<grid_dim, impl::block_dim()>>>
            ( rhs.data(), d.data(), u.data(), parent_index.data(), matrix_sizes.data(),
              padded_matrix_size(), num_matrices());

        // copy the solution from interleaved to front end storage
        reverse_interleave<value_type, size_type, impl::block_dim(), impl::load_width()>
            ( rhs.data(), solution.data(), matrix_sizes.data(), matrix_index.data(),
              padded_matrix_size(), num_matrices());
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest
