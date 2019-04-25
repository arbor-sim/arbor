#pragma once

#include <cstdint>
#include "cuda_atomic.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

// run_length stores information about a run length.
//
// A run length is a set of identical adjacent indexes in an index array.
//
// When doing a parallel reduction by index each thread must know about
// which of its neighbour threads are contributiing to the same global
// location (i.e. which neighbours have the same index).
//
// The constructor takes the thread id and index of each thread
// and the threads work cooperatively using warp shuffles to determine
// their run length information, so that each thread will have unique
// information that describes its run length and its position therein.
struct run_length {
    unsigned width;         // distance to one past the end of this run
    unsigned lane_id;       // id of this warp lane
    unsigned key_mask;      // warp mask of threads participating in reduction
    unsigned is_root;       // if this lane is the first in the run

    __device__
    run_length(int idx, unsigned mask) {
        key_mask = mask;
        lane_id = threadIdx.x%impl::threads_per_warp();
        unsigned num_lanes = impl::threads_per_warp()-__clz(key_mask);

        // Determine if this thread is the root (i.e. first thread with this key).
        int left_idx  = __shfl_up_sync(key_mask, idx, lane_id? 1: 0);

        is_root = lane_id? left_idx!=idx: 1;

        // Determine the range this thread contributes to.
        unsigned roots = __ballot_sync(key_mask, is_root);

        // Find the distance to the lane id one past the end of the run.
        // Take care if this is the last run in the warp.
        width = __ffs(roots>>(lane_id+1));
        if (!width) width = num_lanes-lane_id;
    }
};

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I i, unsigned mask) {
    run_length run(i, mask);
    unsigned shift = 1;
    const unsigned width = run.width;

    unsigned w = shift<width? shift: 0;

    while (__any_sync(run.key_mask, w)) {
        T source_value = __shfl_down_sync(run.key_mask, contribution, w);

        if (w) contribution += source_value;

        shift <<= 1;
        w = shift<width? shift: 0;
    }

    if(run.is_root) {
        // The update must be atomic, because the run may span multiple warps.
        cuda_atomic_add(target+i, contribution);
    }
}

} // namespace gpu
} // namespace arb
