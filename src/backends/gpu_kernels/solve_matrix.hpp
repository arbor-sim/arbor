#pragma once

#include "detail.hpp"

namespace nest {
namespace mc {
namespace gpu {

/// GPU implementation of Hines Matrix solver.
/// Flat format
template <typename T, typename I>
__global__
void solve_matrix_flat(
    T* rhs, T* d, const T* u, const I* p, const I* cell_index, int num_mtx)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if (tid<num_mtx) {
        // get range of this thread's cell matrix
        const auto first = cell_index[tid];
        const auto last  = cell_index[tid+1];

        // backward sweep
        for(auto i=last-1; i>first; --i) {
            auto factor = u[i] / d[i];
            d[p[i]]   -= factor * u[i];
            rhs[p[i]] -= factor * rhs[i];
        }
        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+1; i<last; ++i) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

/// GPU implementation of Hines Matrix solver.
/// Block-interleaved format
template <typename T, typename I, int BlockWidth>
__global__
void solve_matrix_interleaved(
    T* rhs, T* d, const T* u, const I* p, const I* sizes,
    int padded_size, int num_mtx)
{
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < num_mtx) {
        const auto block       = tid/BlockWidth;
        const auto block_start = block*BlockWidth;
        const auto block_lane  = tid - block_start;

        // get range of this thread's cell matrix
        const auto first    = block_start*padded_size + block_lane;
        const auto last     = first + BlockWidth*(sizes[tid]-1);
        const auto last_max = first + BlockWidth*(sizes[block_start]-1);

        // backward sweep
        for(auto i=last_max; i>first; i-=BlockWidth) {
            if (i<=last) {
                auto factor = u[i] / d[i];
                d[p[i]]   -= factor * u[i];
                rhs[p[i]] -= factor * rhs[i];
            }
        }
        rhs[first] /= d[first];

        // forward sweep
        for(auto i=first+BlockWidth; i<=last; i+=BlockWidth) {
            rhs[i] -= u[i] * rhs[p[i]];
            rhs[i] /= d[i];
        }
    }
}

} // namespace gpu
} // namespace mc
} // namespace nest
