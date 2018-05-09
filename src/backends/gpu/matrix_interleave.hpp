#pragma once

#include "cuda_common.hpp"
#include "matrix_common.hpp"

namespace arb {
namespace gpu {

///////////////////////////////////////////////////////////////////////////////
// For more information about the interleaved and flat storage formats for
// Hines matrices, see src/backends/matrix_storage.md
///////////////////////////////////////////////////////////////////////////////

namespace kernels {
// Data in a vector is to be interleaved into blocks of width BlockWidth.
// The kernel assigns LoadWidth threads to each lane in the block.
//
// Note that all indexes can reasonably be represented by an unsigned 32-bit
// integer, so we use unsigned explicitly.
template <typename T, typename I, unsigned BlockWidth, unsigned LoadWidth, unsigned Threads>
__global__
void flat_to_interleaved(
    const T* in, T* out, const I* sizes, const I* starts, unsigned padded_size, unsigned num_vec)
{
    static_assert(BlockWidth*LoadWidth==Threads, "");

    __shared__ T buffer[Threads];

    const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned lid = threadIdx.x;

    const unsigned mtx_id   = tid/LoadWidth;
    const unsigned mtx_lane = tid - mtx_id*LoadWidth;

    const unsigned blk_id   = tid/(BlockWidth*LoadWidth);
    const unsigned blk_row  = lid/BlockWidth;
    const unsigned blk_lane = lid - blk_row*BlockWidth;

    const unsigned blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_load  = mtx_id<num_vec;

    // only threads that participate in loading access starts and sizes arrays
    unsigned load_pos  = do_load? starts[mtx_id] + mtx_lane     : 0u;
    const unsigned end = do_load? starts[mtx_id] + sizes[mtx_id]: 0u;
    unsigned store_pos = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (unsigned i=0u; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (do_load && load_pos<end) {
            loaded = in[load_pos];
        }
        buffer[lid] = loaded;
        __syncthreads();
        if (i+blk_row<padded_size) {
            out[store_pos] = buffer[blk_pos];
        }
        __syncthreads();
        load_pos  += LoadWidth;
        store_pos += LoadWidth*BlockWidth;
    }
}

// Note that all indexes can reasonably be represented by an unsigned 32-bit
// integer, so we use unsigned explicitly.
template <typename T, typename I, unsigned BlockWidth, unsigned LoadWidth, unsigned THREADS>
__global__
void interleaved_to_flat(
    const T* in, T* out, const I* sizes, const I* starts, unsigned padded_size, unsigned num_vec)
{
    static_assert(BlockWidth*LoadWidth==THREADS, "");

    __shared__ T buffer[THREADS];

    const unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;
    const unsigned lid = threadIdx.x;

    const unsigned mtx_id   = tid/LoadWidth;
    const unsigned mtx_lane = tid - mtx_id*LoadWidth;

    const unsigned blk_id   = tid/(BlockWidth*LoadWidth);
    const unsigned blk_row  = lid/BlockWidth;
    const unsigned blk_lane = lid - blk_row*BlockWidth;

    const unsigned blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_store = mtx_id<num_vec;

    // only threads that participate in storing access starts and sizes arrays
    unsigned store_pos = do_store? starts[mtx_id] + mtx_lane     : 0u;
    const unsigned end = do_store? starts[mtx_id] + sizes[mtx_id]: 0u;
    unsigned load_pos  = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (unsigned i=0u; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (i+blk_row<padded_size) {
            loaded = in[load_pos];
        }
        buffer[blk_pos] = loaded;
        __syncthreads();
        if (do_store && store_pos<end) {
            out[store_pos] = buffer[lid];
        }
        __syncthreads();
        load_pos  += LoadWidth*BlockWidth;
        store_pos += LoadWidth;
    }
}

} // namespace kernels

// host side wrapper for the flat to interleaved operation
template <typename T, typename I, unsigned BlockWidth, unsigned LoadWidth>
void flat_to_interleaved(
    const T* in,
    T* out,
    const I* sizes,
    const I* starts,
    unsigned padded_size,
    unsigned num_vec)
{
    constexpr unsigned Threads = BlockWidth*LoadWidth;
    const unsigned blocks = impl::block_count(num_vec, BlockWidth);

    kernels::flat_to_interleaved
        <T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>>
        (in, out, sizes, starts, padded_size, num_vec);
}

// host side wrapper for the interleave to flat operation
template <typename T, typename I, unsigned BlockWidth, unsigned LoadWidth>
void interleaved_to_flat(
    const T* in,
    T* out,
    const I* sizes,
    const I* starts,
    unsigned padded_size,
    unsigned num_vec)
{
    constexpr unsigned Threads = BlockWidth*LoadWidth;
    const unsigned blocks = impl::block_count(num_vec, BlockWidth);

    kernels::interleaved_to_flat
        <T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>>
        (in, out, sizes, starts, padded_size, num_vec);
}

} // namespace gpu
} // namespace arb

