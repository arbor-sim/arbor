#include <arbor/fvm_types.hpp>

#include "gpu_api.hpp"
#include "gpu_common.hpp"
#include "matrix_common.hpp"
#include "matrix_fine.hpp"

namespace arb {
namespace gpu {

namespace kernels {

//
// gather and scatter kernels
//

// to[i] = from[p[i]]
template <typename T, typename I>
__global__
void gather(const T* from, T* to, const I* p, unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) {
        to[i] = from[p[i]];
    }
}

// to[p[i]] = from[i]
template <typename T, typename I>
__global__
void scatter(const T* from, T* to, const I* p, unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) {
        to[p[i]] = from[i];
    }
}

/// GPU implementation of Hines matrix assembly.
/// Fine layout.
/// For a given time step size dt:
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve.
template <typename T, typename I>
__global__
void assemble_matrix_fine(
        T* __restrict__ d,
        T* __restrict__ rhs,
        const T* __restrict__ const invariant_d,
        const T* __restrict__ const voltage,
        const T* __restrict__ const current,
        const T* __restrict__ const conductivity,
        const T* __restrict__ const cv_capacitance,
        const T* __restrict__ const area,
        const I* __restrict__ const cv_to_intdom,
        const T* __restrict__ const dt_intdom,
        const I* __restrict__ const perm,
        unsigned n)
{
    const unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid < n) {
        // The 1e-3 is a constant of proportionality required to ensure that the
        // conductance (gi) values have units μS (micro-Siemens).
        // See the model documentation in docs/model for more information.
        const auto dt = dt_intdom[cv_to_intdom[tid]];
        const auto p = dt > 0;
        const auto pid = perm[tid];
        const auto u = voltage[tid];
        const auto area_factor = T(1e-3)*area[tid];
        const auto gi = T(1e-3)*cv_capacitance[tid]/dt + area_factor*conductivity[tid];
        const auto r_d = gi + invariant_d[tid];
        const auto r_rhs = gi*u - area_factor*current[tid];

        d[pid]   = p ? r_d : 0;
        rhs[pid] = p ? r_rhs : u;
    }
}

/// GPU implementation of Hines Matrix solver.
/// Fine-grained tree based solver.
/// Each block solves a set of matricesb iterating over the levels of matrix
/// and perfoming a backward and forward substitution. On each level one thread
/// gets assigned to one branch on this level of a matrix and solves and
/// performs the substitution. Afterwards all threads continue on the next
/// level.
/// To avoid idle threads, one should try that on each level, there is a similar
/// number of branches.
template <typename T>
__global__
void solve_matrix_fine(
    T* __restrict__ rhs,
    T* __restrict__ d,
    const T* __restrict__ u,
    const level_metadata* __restrict__ level_meta,
    const fvm_index_type* __restrict__ level_lengths,
    const fvm_index_type* __restrict__ level_parents,
    const fvm_index_type* __restrict__ block_index,
    const fvm_index_type* __restrict__ num_matrix) // number of packed matrices = number of cells
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    const auto first_level = block_index[bid];
    const auto num_levels  = block_index[bid + 1] - first_level;

    const auto block_level_meta = &level_meta[first_level];

    // backward substitution

    for (unsigned l=0; l<num_levels-1; ++l) {
        // Metadata for this level and the next level
        const auto& lvl_meta = block_level_meta[l];

        // Addresses of the first elements of level_lengths and level_parents
        // that belong to this level
        const auto lvl_lengths = level_lengths + lvl_meta.level_data_index;
        const auto lvl_parents = level_parents + lvl_meta.level_data_index;
        const auto lvl_data_index = lvl_meta.matrix_data_index;
        
        const unsigned width = lvl_meta.num_branches;

        const auto parent_index = block_level_meta[l+1].matrix_data_index;

        // Perform backward substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            // Indices delimiting this level
            const unsigned lvl_data_beg = lvl_data_index + tid;
            const unsigned lvl_data_end = lvl_data_beg + (lvl_lengths[tid] - 1)*width;
            // Index of parent
            const unsigned p = parent_index + lvl_parents[tid];
            // Zero diagonal term implies dt==0; just leave rhs (for whole matrix)
            // alone in that case.

            // Each cell has a different `dt`, because we choose time step size
            // according to when the next event is arriving at a cell. So, some
            // cells require more time steps than others, but we have to solve
            // all the matrices at the same time. When a cell finishes, we put a
            // `0` on the diagonal to mark that it should not be solved for.
            if (d[lvl_data_beg] != 0) {
                // each branch perform substitution
                for (unsigned pos = lvl_data_beg; pos < lvl_data_end; pos += width) {
                    const auto pos_next = pos + width;
                    const auto u_ = u[pos];
                    const auto d_next = d[pos_next];                    
                    const auto rhs_ = rhs[pos];
                    const auto rhs_next = rhs[pos_next];
                    const T factor = -u_/d[pos];
                    d[pos_next]   = fma(factor, u_, d_next);
                    rhs[pos_next] = fma(factor, rhs_, rhs_next);
                }
                // Update d and rhs at the parent node of this branch.
                // A parent may have more than one contributing to it, so we use
                // atomic updates to avoid races conditions.
                const auto u_ = u[lvl_data_end];
                const auto d_ = d[lvl_data_end];
                const auto rhs_ = rhs[lvl_data_end];
                const T factor = -u_[lvl_data_end] / d_[lvl_data_end];
                gpu_atomic_add(d + p,  factor*u_);
                gpu_atomic_add(rhs + p, factor*rhs_);
            }
        }
        __syncthreads();
    }

    // Solve the root
    {
        // The levels are sorted such that the root is the last level
        const auto& last_lvl_meta = block_level_meta[num_levels-1];
        const auto lvl_data_index = last_lvl_meta.matrix_data_index;
        const auto lvl_lengths = level_lengths + last_lvl_meta.level_data_index;

        const unsigned width = num_matrix[bid];

        if (tid < width) {
            const int lvl_data_beg = lvl_data_index + tid;
            const int lvl_data_end = lvl_data_beg + (lvl_lengths[tid] - 1)*width;
            if (d[lvl_data_beg]!=0) {
                // backward
                for (int pos = lvl_data_beg; pos < lvl_data_end; pos += width) {
                    const auto pos_next = pos + width;
                    const auto u_ = u[pos];
                    const auto rhs_ = rhs[pos];
                    const T factor = -u_ / d[pos];
                    const auto rhs_next = rhs[pos_next];
                    const auto d_next = d[pos_next];
                    d[pos_next]   = fma(factor, u_, d_next);
                    rhs[pos_next] = fma(factor, rhs_, rhs_next);
                }

                // forward
                T rhsp = 0.0;                
                for (int pos = lvl_data_end; pos >= lvl_data_beg; pos -= width) {
                    rhsp = (rhs[pos] - u[pos]*rhsp)/d[pos];
                    rhs[pos] = rhsp;
                }
            }
        }
    }

    // forward substitution

    // take great care with loop limits decrementing unsigned counter l
    for (unsigned l=num_levels-1; l>0; --l) {
        const auto& lvl_meta = block_level_meta[l-1];

        // Addresses of the first elements of level_lengths and level_parents
        // that belong to this level
        const auto lvl_lengths = level_lengths + lvl_meta.level_data_index;
        const auto lvl_parents = level_parents + lvl_meta.level_data_index;
        const auto lvl_data_index = lvl_meta.matrix_data_index;
        const unsigned width = lvl_meta.num_branches;
        const unsigned parent_index = block_level_meta[l].matrix_data_index;

        __syncthreads();

        // Perform forward-substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            const unsigned p = parent_index + lvl_parents[tid];
            // Find the index of the first node in this branch.
            const int lvl_data_beg = lvl_data_index + tid;
            const int lvl_data_end = lvl_data_beg + (lvl_lengths[tid] - 1)*width;

            if (d[lvl_data_end]!=0) {
                // Load the rhs value for the parent node of this branch.
                T rhsp = rhs[p];
                // each branch perform substitution
                for (int pos = lvl_data_end; pos >= lvl_data_beg; pos -= width) {
                    const auto u_ = u[pos];
                    const auto d_ = d[pos];
                    const auto rhs_ = rhs[pos];
                    rhsp = (rhs_ - u_*rhsp)/d_;
                    rhs[pos] = rhsp;
                }
            }
        }
    }
}

} // namespace kernels

void gather(
    const fvm_value_type* from,
    fvm_value_type* to,
    const fvm_index_type* p,
    unsigned n)
{
    constexpr unsigned blockdim = 128;
    const unsigned griddim = impl::block_count(n, blockdim);

    kernels::gather<<<griddim, blockdim>>>(from, to, p, n);
}

void scatter(
    const fvm_value_type* from,
    fvm_value_type* to,
    const fvm_index_type* p,
    unsigned n)
{
    constexpr unsigned blockdim = 128;
    const unsigned griddim = impl::block_count(n, blockdim);

    kernels::scatter<<<griddim, blockdim>>>(from, to, p, n);
}

void assemble_matrix_fine(
    fvm_value_type* d,
    fvm_value_type* rhs,
    const fvm_value_type* invariant_d,
    const fvm_value_type* voltage,
    const fvm_value_type* current,
    const fvm_value_type* conductivity,
    const fvm_value_type* cv_capacitance,
    const fvm_value_type* area,
    const fvm_index_type* cv_to_intdom,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* perm,
    unsigned n)
{
    const unsigned block_dim = 128;
    const unsigned num_blocks = impl::block_count(n, block_dim);

    kernels::assemble_matrix_fine<<<num_blocks, block_dim>>>(
        d, rhs, invariant_d, voltage, current, conductivity, cv_capacitance, area,
        cv_to_intdom, dt_intdom, perm, n);
}

// Example:
//
//         block 0                  block 1              block 2
// .~~~~~~~~~~~~~~~~~~.  .~~~~~~~~~~~~~~~~~~~~~~~~.  .~~~~~~~~~~~ ~ ~
//
//  L0 \  /                                           L5    \  /
//      \/                                                   \/
//  L1   \   /   \   /    L3 \   /   \ | /   \   /    L6 \   /  . . .
//        \ /     \ /         \ /     \|/     \ /         \ /
//  L2     |       |      L4   |       |       |      L7   |
//         |       |           |       |       |           |
//
// levels       = [L0, L1, L2, L3, L4, L5, L6, L7, ... ]
// block_index  = [0, 3, 5, 8, ...]
// num_levels   = [3, 2, 3, ...]
// num_cells    = [2, 3, ...]
// num_blocks   = level_start.size() - 1 = num_levels.size() = num_cells.size()
void solve_matrix_fine(
    fvm_value_type* rhs,
    fvm_value_type* d,                     // diagonal values
    const fvm_value_type* u,               // upper diagonal (and lower diagonal as the matrix is SPD)
    const level_metadata* level_meta,      // information pertaining to each level
    const fvm_index_type* level_lengths,   // lengths of branches of every level concatenated
    const fvm_index_type* level_parents,   // parents of branches of every level concatenated
    const fvm_index_type* block_index,     // start index into levels for each gpu block
    fvm_index_type* num_cells,             // the number of cells packed into this single matrix
    fvm_index_type* padded_size,           // length of rhs, d, u, including padding
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize)                    // size of each block
{
    kernels::solve_matrix_fine<<<num_blocks, blocksize>>>(
        rhs, d, u, level_meta, level_lengths, level_parents, block_index,
        num_cells);
}

} // namespace gpu
} // namespace arb
