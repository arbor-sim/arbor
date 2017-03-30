#pragma once

#include "detail.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// GPU implementatin of Hines matrix assembly
/// Flat layout
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I>
__global__
void assemble_matrix_flat(
        T* d, T* rhs, const T* invariant_d,
        const T* voltage, const T* current, const T* cv_capacitance,
        T dt, int n)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    T factor = 1e-3/dt;
    if (tid<n) {
        auto gi = factor * cv_capacitance[tid];
        d[tid] = gi + invariant_d[tid];
        rhs[tid] = gi*voltage[tid] - current[tid];
    }
}

/// GPU implementatin of Hines matrix assembly
/// Interleaved layout
/// For a given time step size dt
///     - use the precomputed alpha and alpha_d values to construct the diagonal
///       and off diagonal of the symmetric Hines matrix.
///     - compute the RHS of the linear system to solve
template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void assemble_matrix_interleaved(
        T* d,
        T* rhs,
        const T* invariant_d,
        const T* voltage,
        const T* current,
        const T* cv_capacitance,
        const I* sizes,
        const I* starts,
        T dt, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==Threads,
        "number of threads must equal number of values to process per block");
    __shared__ T buffer_v[Threads];
    __shared__ T buffer_i[Threads];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_load  = mtx_id<num_mtx;

    auto load_pos  = do_load? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_load? starts[mtx_id] + sizes[mtx_id]: 0;
    auto store_pos = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    const auto max_size = sizes[0];

    T factor = 1e-3/dt;
    for (auto j=0; j<max_size; j+=LoadWidth) {
        if (do_load && load_pos<end) {
            buffer_v[lid] = voltage[load_pos];
            buffer_i[lid] = current[load_pos];
        }

        __syncthreads();

        if (j+blk_row<padded_size) {
            const auto gi = factor * cv_capacitance[store_pos];
            d[store_pos]   = gi + invariant_d[store_pos];
            rhs[store_pos] = gi*buffer_v[blk_pos] - buffer_i[blk_pos];
        }

        store_pos += LoadWidth*BlockWidth;
        load_pos  += LoadWidth;
    }
}

} // namespace gpu
} // namespace mc
} // namespace nest
