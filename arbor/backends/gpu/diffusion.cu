#include <arbor/fvm_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "matrix_common.hpp"
#include "diffusion.hpp"

namespace arb {
namespace gpu {
namespace kernels {
/// GPU implementation of Hines matrix assembly.
/// Fine layout.
/// For a given time step size dt:
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve.
template <typename T, typename I>
__global__
void assemble_diffusion(
        T* __restrict__ const d,
        T* __restrict__ const rhs,
        const T* __restrict__ const invariant_d,
        const T* __restrict__ const concentration,
        const T* __restrict__ const voltage,
        const T* __restrict__ const current,
        const T q,
        const T* __restrict__ const conductivity,
        const T* __restrict__ const area,
        const T* __restrict__ const volume,
        const T dt,
        const I* __restrict__ const perm,
        unsigned n) {
    const unsigned tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid < n) {
        const auto pid = perm[tid];
        auto u = voltage[tid];        // mV
        auto g = conductivity[tid];   // µS
        auto J = current[tid];        // A/m^2
        auto A = 1e-3*area[tid];      // 1e-9·m²
        auto V = volume[tid];         // um^3
        auto X = concentration[tid];  // mM
        // conversion from current density to concentration change
        // using Faraday's constant
        auto F = A/(q*96.485332);

        d[pid]   = 1e-3*V/dt   + F*g + invariant_d[tid];
        rhs[pid] = 1e-3*V/dt*X + F*(u*g - J);
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
void solve_diffusion(
    T* __restrict__ const rhs,
    T* __restrict__ const d,
    const T* __restrict__ const u,
    const level_metadata* __restrict__ const level_meta,
    const arb_index_type* __restrict__ const level_lengths,
    const arb_index_type* __restrict__ const level_parents,
    const arb_index_type* __restrict__ const block_index,
    const arb_index_type* __restrict__ const num_matrix) // number of packed matrices = number of cells
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

            // each branch perform substitution
            for (unsigned i=0; i<len-1; ++i) {
                const unsigned next_pos = pos + width;
                const auto d_next = d[next_pos];
                const auto rhs_next = rhs[next_pos];
                const T factor = -u[pos]/d[pos];
                d[next_pos] = fma(factor, u[pos], d_next);
                rhs[next_pos] = fma(factor, rhs[pos], rhs_next);
                pos = next_pos;
            }
            // Update d and rhs at the parent node of this branch.
            // A parent may have more than one contributing to it, so we use
            // atomic updates to avoid races conditions.
            const unsigned parent_index = next_lvl_meta.matrix_data_index;
            const unsigned p = parent_index + lvl_parents[tid];

            const T factor = -u[pos] / d[pos];
            gpu_atomic_add(d + p, factor*u[pos]);
            gpu_atomic_add(rhs + p, factor*rhs[pos]);
        }
        __syncthreads();
    }

    // Solve the root
    {
        // The levels are sorted such that the root is the last level
        const auto& last_lvl_meta = block_level_meta[num_levels-1];
        const auto lvl_lengths = level_lengths + last_lvl_meta.level_data_index;

        const unsigned width = num_matrix[bid];

        if (tid < width) {
            const unsigned len = lvl_lengths[tid];
            unsigned pos = last_lvl_meta.matrix_data_index + tid;

            // backward
            for (unsigned i=0; i<len-1; ++i) {
                const unsigned next_pos = pos + width;
                const T factor = -u[pos] / d[pos];
                const auto rhs_next = rhs[next_pos];
                const auto d_next = d[next_pos];
                d[next_pos]   = fma(factor, u[pos], d_next);
                rhs[next_pos] = fma(factor, rhs[pos], rhs_next);
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
            // Find the index of the first node in this branch.
            const unsigned len = lvl_lengths[tid];
            unsigned pos = lvl_meta.matrix_data_index + (len-1)*width + tid;

            // Load the rhs value for the parent node of this branch.
            const unsigned p = parent_index + lvl_parents[tid];
            T rhsp = rhs[p];
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

} // namespace kernels

ARB_ARBOR_API void assemble_diffusion(
    arb_value_type* d,
    arb_value_type* rhs,
    const arb_value_type* invariant_d,
    const arb_value_type* concentration,
    const arb_value_type* voltage,
    const arb_value_type* current,
    arb_value_type q,
    const arb_value_type* conductivity,
    const arb_value_type* area,
    const arb_value_type* volume,
    const arb_value_type dt,
    const arb_index_type* perm,
    unsigned n)
{
    launch_1d(n, 128, kernels::assemble_diffusion<arb_value_type, arb_index_type>,
        d, rhs, invariant_d, concentration, voltage, current, q, conductivity, area, volume
        dt, perm, n);
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
ARB_ARBOR_API void solve_diffusion(
    arb_value_type* rhs,
    arb_value_type* d,                     // diagonal values
    const arb_value_type* u,               // upper diagonal (and lower diagonal as the matrix is SPD)
    const level_metadata* level_meta,      // information pertaining to each level
    const arb_index_type* level_lengths,   // lengths of branches of every level concatenated
    const arb_index_type* level_parents,   // parents of branches of every level concatenated
    const arb_index_type* block_index,     // start index into levels for each gpu block
    arb_index_type* num_cells,             // the number of cells packed into this single matrix
    arb_index_type* padded_size,           // length of rhs, d, u, including padding
    unsigned num_blocks,                   // number of blocks
    unsigned blocksize)                    // size of each block
{
    launch(num_blocks, blocksize, kernels::solve_diffusion<arb_value_type>,
        rhs, d, u, level_meta, level_lengths, level_parents, block_index,
        num_cells);
}

} // namespace gpu
} // namespace arb
