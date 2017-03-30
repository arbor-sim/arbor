#pragma once

#include "detail.hpp"

namespace nest {
namespace mc {
namespace gpu {

// Data is to be interleaved into blocks of width BlockWidth.
// The kernel assigns LoadWidth threads to each lane in the block.
// Hence each thread block is responsible for loading a single block
// of interleaved matrices.
template <typename T, typename I, int BlockWidth, int LoadWidth, int Threads>
__global__
void interleave(
    const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==Threads, "");

    __shared__ T buffer[Threads];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_load  = mtx_id<num_mtx;

    // only threads that participate in loading access starts and sizes arrays
    auto load_pos  = do_load? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_load? starts[mtx_id] + sizes[mtx_id]: 0;
    auto store_pos = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (auto i=0; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (do_load && load_pos<end) {
            loaded = in[load_pos];
        }
        buffer[lid] = loaded;
        __syncthreads();
        if (i+blk_row<padded_size) {
            out[store_pos] = buffer[blk_pos];
        }
        load_pos  += LoadWidth;
        store_pos += LoadWidth*BlockWidth;
    }
}

template <typename T, typename I, int BlockWidth, int LoadWidth, int THREADS>
__global__
void reverse_interleave(
    const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    static_assert(BlockWidth*LoadWidth==THREADS, "");

    __shared__ T buffer[THREADS];

    const auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    const auto lid = threadIdx.x;

    const auto mtx_id   = tid/LoadWidth;
    const auto mtx_lane = tid - mtx_id*LoadWidth;

    const auto blk_id   = tid/(BlockWidth*LoadWidth);
    const auto blk_row  = lid/BlockWidth;
    const auto blk_lane = lid - blk_row*BlockWidth;

    const auto blk_pos  = LoadWidth*blk_lane + blk_row;

    const bool do_store = mtx_id<num_mtx;

    // only threads that participate in storing access starts and sizes arrays
    auto store_pos = do_store? starts[mtx_id] + mtx_lane     : 0;
    const auto end = do_store? starts[mtx_id] + sizes[mtx_id]: 0;
    auto load_pos  = blk_id*BlockWidth*padded_size + (blk_row*BlockWidth + blk_lane);

    for (auto i=0; i<padded_size; i+=LoadWidth) {
        auto loaded = impl::npos<T>();
        if (i+blk_row<padded_size) {
            loaded = in[load_pos];
        }
        buffer[blk_pos] = loaded;
        __syncthreads();
        if (do_store && store_pos<end) {
            out[store_pos] = buffer[lid];
        }
        load_pos  += LoadWidth*BlockWidth;
        store_pos += LoadWidth;
    }
}

// host side wrapper for the flat to interleaved operation
template <typename T, typename I, int BlockWidth, int LoadWidth>
void interleave(const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    constexpr int Threads = BlockWidth*LoadWidth;
    const int blocks = impl::block_count(num_mtx, BlockWidth);

    interleave<T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>> (in, out, sizes, starts, padded_size, num_mtx);
}

// host side wrapper for the interleave to flat operation
template <typename T, typename I, int BlockWidth, int LoadWidth>
void reverse_interleave(const T* in, T* out, const I* sizes, const I* starts, int padded_size, int num_mtx)
{
    constexpr int Threads = BlockWidth*LoadWidth;
    const int blocks = impl::block_count(num_mtx, BlockWidth);

    reverse_interleave<T, I, BlockWidth, LoadWidth, Threads>
        <<<blocks, Threads>>> (in, out, sizes, starts, padded_size, num_mtx);
}


} // namespace gpu
} // namespace mc
} // namespace nest
