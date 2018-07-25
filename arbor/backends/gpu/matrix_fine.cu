#include <arbor/fvm_types.hpp>

#include "cuda_atomic.hpp"
#include "cuda_common.hpp"
#include "matrix_common.hpp"
#include "matrix_fine.hpp"

namespace arb {
namespace gpu {

namespace kernels {

/// GPU implementation of Hines Matrix solver.
/// Fine-grained tree based solver.
template <typename T>
__global__
void solve_matrix_fine(
    T* rhs,
    T* d,
    const T* u,
    const level* levels,
    unsigned num_matrix,
    unsigned num_levels,
    unsigned padded_size)
{
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;

    // backward substitution

    for (unsigned l=0; l<num_levels-1; ++l) {
        const auto& lvl = levels[l];

        const unsigned width = lvl.num_branches;

        // Perform backward substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            const unsigned len = lvl.lengths[tid];
            unsigned pos = lvl.data_index + tid;

            // Zero diagonal term implies dt==0; just leave rhs (for whole matrix)
            // alone in that case.
            if (d[pos]==0) return;

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
            const unsigned parent_index = levels[l+1].data_index;
            const unsigned p = parent_index + lvl.parents[tid];
            //d[p]   -= factor * u[pos];
            cuda_atomic_add(d  +p, -factor*u[pos]);
            //rhs[p] -= factor * rhs[pos];
            cuda_atomic_add(rhs+p, -factor*rhs[pos]);
        }
        __syncthreads();
    }

    // take advantage of the fact that the last num_matrix in rhs and d
    // correspond to the root nodes.
    if (tid<num_matrix) {
        unsigned pos = padded_size - num_matrix + tid;
        rhs[pos] /= d[pos];
    }

    // forward substitution

    // take great care with loop limits decrementing unsigned counter l
    for (unsigned l=num_levels-1; l>0; --l) {
        const auto& lvl = levels[l-1];

        const unsigned width = lvl.num_branches;
        const unsigned parent_index = levels[l].data_index;

        // Perform forward-substitution for each branch on this level.
        // One thread per branch.
        if (tid < width) {
            // Load the rhs value for the parent node of this branch.
            const unsigned p = parent_index + lvl.parents[tid];
            T rhsp = rhs[p];

            // Find the index of the first node in this branch.
            const unsigned len = lvl.lengths[tid];
            unsigned pos = lvl.data_index + (len-1)*width + tid;

            for (unsigned i=0; i<len; ++i) {
                rhsp = rhs[pos] - u[pos]*rhsp;
                rhsp /= d[pos];
                rhs[pos] = rhsp;
                pos -= width;
            }
        }
        __syncthreads();
    }
}

} // namespace kernels

void solve_matrix_fine(
    fvm_value_type* rhs,
    fvm_value_type* d,
    const fvm_value_type* u,
    const level* levels,              // pointer to an array containing level meta-data
    unsigned num_cells,               // the number of cells packed into this single matrix
    unsigned num_levels,              // depth of the tree (in branches)
    unsigned padded_size,             // length of rhs, d, u, including padding
    unsigned max_branches_per_level)  // the maximum number of branches on any level
{
    kernels::solve_matrix_fine<<<1, max_branches_per_level>>>(
        rhs, d, u, levels,
        num_cells, num_levels, padded_size);
}

} // namespace gpu
} // namespace arb
