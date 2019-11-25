#include <arbor/fvm_types.hpp>

#include "cuda_atomic.hpp"
#include "cuda_common.hpp"
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

/// GPU implementatin of Hines matrix assembly.
/// Fine layout.
/// For a given time step size dt:
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve.
template <typename T, typename I>
__global__
void assemble_matrix_fine(
        T* d,
        T* rhs,
        const T* invariant_d,
        const T* voltage,
        const T* current,
        const T* conductivity,
        const T* cv_capacitance,
        const T* area,
        const I* cv_to_cell,
        const T* dt_intdom,
        const I* cell_to_intdom,
        const I* perm,
        unsigned n)
{
    const unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid<n) {
        auto cid = cv_to_cell[tid];
        auto dt = dt_intdom[cell_to_intdom[cid]];

        if (dt>0) {
            // The 1e-3 is a constant of proportionality required to ensure that the
            // conductance (gi) values have units μS (micro-Siemens).
            // See the model documentation in docs/model for more information.
            T oodt_factor = T(1e-3)/dt;
            T area_factor = T(1e-3)*area[tid];

            const auto gi = oodt_factor*cv_capacitance[tid] + area_factor*conductivity[tid];
            const auto pid = perm[tid];
            d[pid] = gi + invariant_d[tid];
            rhs[pid] = gi*voltage[tid] - area_factor*current[tid];
        }
        else {
            const auto pid = perm[tid];
            d[pid] = 0;
            rhs[pid] = voltage[tid];
        }
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
    T* rhs,
    T* d,
    const T* u,
    const level_metadata* level_meta,
    const fvm_index_type* level_lengths,
    const fvm_index_type* level_parents,
    const fvm_index_type* block_index,
    fvm_index_type* num_matrix, // number of packed matrices = number of cells
    fvm_index_type* padded_size)
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
        const auto& next_lvl_meta = block_level_meta[l+1];

        // Addresses of the first elements of level_lengths and level_parents
        // that belong to this level
        const auto lvl_lengths = level_lengths + lvl_meta.level_data_index;
        const auto lvl_parents = level_parents + lvl_meta.level_data_index;

        const unsigned width = lvl_meta.num_branches;

        // Perform backward substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            const unsigned len = lvl_lengths[tid];
            unsigned pos = lvl_meta.matrix_data_index + tid;

            // Zero diagonal term implies dt==0; just leave rhs (for whole matrix)
            // alone in that case.

            // Each cell has a different `dt`, because we choose time step size
            // according to when the next event is arriving at a cell. So, some
            // cells require more time steps than others, but we have to solve
            // all the matrices at the same time. When a cell finishes, we put a
            // `0` on the diagonal to mark that it should not be solved for.
            if (d[pos]!=0) {

                // each branch perform substitution
                T factor = u[pos] / d[pos];
                for (unsigned i=0; i<len-1; ++i) {
                    const unsigned next_pos = pos + width;
                    d[next_pos]   -= factor * u[pos];
                    rhs[next_pos] -= factor * rhs[pos];

                    factor = u[next_pos] / d[next_pos];
                    pos = next_pos;
                }

                // Update d and rhs at the parent node of this branch.
                // A parent may have more than one contributing to it, so we use
                // atomic updates to avoid races conditions.
                const unsigned parent_index = next_lvl_meta.matrix_data_index;
                const unsigned p = parent_index + lvl_parents[tid];
                //d[p]   -= factor * u[pos];
                cuda_atomic_add(d  +p, -factor*u[pos]);
                //rhs[p] -= factor * rhs[pos];
                cuda_atomic_add(rhs+p, -factor*rhs[pos]);
            }
        }
        __syncthreads();
    }

    {
        // The levels are sorted such that the root is the last level
        const auto& last_lvl_meta = block_level_meta[num_levels-1];
        const auto lvl_lengths = level_lengths + last_lvl_meta.level_data_index;

        const unsigned width = num_matrix[bid];

        if (tid < width) {
            const unsigned len = lvl_lengths[tid];
            unsigned pos = last_lvl_meta.matrix_data_index + tid;

            if (d[pos]!=0) {

                // backward
                for (unsigned i=0; i<len-1; ++i) {
                    T factor = u[pos] / d[pos];
                    const unsigned next_pos = pos + width;
                    d[next_pos]   -= factor * u[pos];
                    rhs[next_pos] -= factor * rhs[pos];

                    pos = next_pos;
                }

                auto rhsp = rhs[pos] / d[pos];
                rhs[pos] = rhsp;
                pos -= width;

                // forward
                for (unsigned i=0; i<len-1; ++i) {
                    rhsp = rhs[pos] - u[pos]*rhsp;
                    rhsp /= d[pos];
                    rhs[pos] = rhsp;
                    pos -= width;
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

        const unsigned width = lvl_meta.num_branches;
        const unsigned parent_index = block_level_meta[l].matrix_data_index;

        __syncthreads();

        // Perform forward-substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            // Load the rhs value for the parent node of this branch.
            const unsigned p = parent_index + lvl_parents[tid];
            T rhsp = rhs[p];

            // Find the index of the first node in this branch.
            const unsigned len = lvl_lengths[tid];
            unsigned pos = lvl_meta.matrix_data_index + (len-1)*width + tid;

            if (d[pos]!=0) {
                // each branch perform substitution
                for (unsigned i=0; i<len; ++i) {
                    rhsp = rhs[pos] - u[pos]*rhsp;
                    rhsp /= d[pos];
                    rhs[pos] = rhsp;
                    pos -= width;
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
    const fvm_index_type* cv_to_cell,
    const fvm_value_type* dt_intdom,
    const fvm_index_type* cell_to_intdom,
    const fvm_index_type* perm,
    unsigned n)
{
    const unsigned block_dim = 128;
    const unsigned num_blocks = impl::block_count(n, block_dim);

    kernels::assemble_matrix_fine<<<num_blocks, block_dim>>>(
        d, rhs, invariant_d, voltage, current, conductivity, cv_capacitance, area,
        cv_to_cell, dt_intdom, cell_to_intdom,
        perm, n);
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
    const fvm_index_type* block_index,     // start index into levels for each cuda block
    fvm_index_type* num_cells,             // the number of cells packed into this single matrix
    fvm_index_type* padded_size,           // length of rhs, d, u, including padding
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize)                    // size of each block
{
    kernels::solve_matrix_fine<<<num_blocks, blocksize>>>(
        rhs, d, u, level_meta, level_lengths, level_parents, block_index,
        num_cells, padded_size);
}

} // namespace gpu
} // namespace arb
