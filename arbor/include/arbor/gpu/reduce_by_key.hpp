#pragma once

#include <cstdint>
#include "gpu_api.hpp"
#include "gpu_common.hpp"

namespace arb {
namespace gpu {

// Fixed trees over sorted key runs. Launch 128-thread blocks with a power-of-two
// group_width <= 128; divisions includes the final input size.
template <typename T, typename I>
__device__ inline
void gather_by_key(const T* contributions, T* target, const I* index, const I* divisions, unsigned n, unsigned group_width) {
    unsigned tid = threadIdx.x;
    unsigned group = blockIdx.x*(blockDim.x/group_width) + tid/group_width;
    unsigned lane = tid%group_width;
    unsigned begin = group<n? divisions[group]: 0;
    unsigned end = group<n? divisions[group+1]: 0;
    __shared__ T partial[128];
    T sum = 0;
    // shortcut: one group per CV; split very long runs if model profiling shows a bottleneck.
    for (unsigned i = begin+lane; i<end; i+=group_width) sum += contributions[i];
    if (group_width==1) {
        if (group<n) target[index[begin]] += sum;
        return;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = group_width/2; stride; stride/=2) {
        if (lane<stride) partial[tid] += partial[tid+stride];
        __syncthreads();
    }
    if (group<n && !lane) target[index[begin]] += partial[tid];
    __syncthreads(); // A following field can reuse the shared storage.
}

// key_set_pos stores information required by a thread to calculate its
// contribution to a reduce by key operation.
//
// In reduce_by_key each thread performs a reduction with threads that have the same
// index in a sorted index array.
// The algorithm for reduce_by_key implemented here requires that each thread
// knows how far it is from the end of the key set, and whether it is the first
// thread in the warp with the key.
//
// As currently implemented, this could be performed inline inside the reduce_by_key
// function, however it is in a separate data type as a first step towards reusing
// the same key set information for multipe reduction kernel calls.

struct key_set_pos {
    unsigned width;         // distance to one past the end of this run
    unsigned lane_id;       // id of this warp lane
    lane_mask_type key_mask; // warp mask of threads participating in reduction
    unsigned is_root;       // if this lane is the first in the run

    // The constructor takes the index of each thread and a mask of which threads
    // in the warp are participating in the reduction, and the threads work cooperatively
    // using warp intrinsics and bit twiddling, so that each thread will have unique
    // information that describes its position in the key set.
    __device__
    key_set_pos(int idx, lane_mask_type mask) {
        key_mask = mask;
        lane_id = threadIdx.x%impl::threads_per_warp();
        // count_leading_zeros counts from the top bit of lane_mask_type, so the
        // lane count is measured against its full bit width: 64 on HIP (ballot is
        // wave64-wide on CDNA, wave32 on RDNA) and 32 on CUDA. On a wave32 HIP
        // mask the upper 32 bits are zero, so 64 - count_leading_zeros still
        // yields the correct count.
        unsigned num_lanes = 8*sizeof(lane_mask_type) - count_leading_zeros(key_mask);

        // Determine if this thread is the root (i.e. first thread with this key).
        int left_idx  = shfl_up(key_mask, idx, lane_id, lane_id? 1: 0);

        is_root = lane_id? left_idx!=idx: 1;

        // Determine the range this thread contributes to.
        lane_mask_type roots = ballot(key_mask, is_root);

        // Find the distance to the lane id one past the end of the run.
        // Take care if this is the last run in the warp.
        width = find_first_set(roots>>(lane_id+1));
        if (!width) width = num_lanes-lane_id;
    }
};

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I i, lane_mask_type mask) {
    key_set_pos run(i, mask);
    unsigned shift = 1;
    const unsigned width = run.width;

    unsigned w = shift<width? shift: 0;

    while (any(run.key_mask, w)) {
        T source_value = shfl_down(run.key_mask, contribution, run.lane_id, w);

        if (w) contribution += source_value;

        shift <<= 1;
        w = shift<width? shift: 0;
    }

    if(run.is_root) {
        // The update must be atomic, because the run may span multiple warps.
        gpu_atomic_add(target+i, contribution);
    }
}

} // namespace gpu
} // namespace arb
