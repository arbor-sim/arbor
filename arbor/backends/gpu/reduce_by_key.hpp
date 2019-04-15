#pragma once

#include <cstdint>
#include "cuda_atomic.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

namespace impl{

constexpr unsigned mask_all = 0xFFFFFFFF;

// return the power of 2 that is less than or equal to i
__device__ __inline__
unsigned rounddown_power_of_2(std::uint32_t i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(31u - __clz(i));
}

// run_length_type Stores information about a run length.
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
struct run_length_type {
    unsigned left;
    unsigned right;
    unsigned width;
    unsigned lane_id;
    unsigned key_mask;

    __device__ __inline__
    bool is_root() const {
        return left == lane_id;
    }

    __device__ __inline__
    bool may_cross_warp() const {
        return left==0 || right==threads_per_warp();
    }

    template <typename I1>
    __device__
    run_length_type(I1 idx) {
        lane_id = threadIdx.x%threads_per_warp();
        __syncwarp();
        key_mask = __activemask();
        unsigned num_lanes = threads_per_warp()-__clz(key_mask);

        auto right_limit = [num_lanes] (unsigned roots, unsigned shift) {
            unsigned zeros_right  = __ffs(roots>>shift);
            return zeros_right ? shift -1 + zeros_right : num_lanes;
        };

        // determine if this thread is the root
        int left_idx  = __shfl_up_sync(key_mask, idx, 1);
        int is_root = 1;
        if(lane_id>0) {
            is_root = (left_idx != idx);
        }

        // determine the range this thread contributes to
        unsigned roots = __ballot_sync(key_mask, is_root);

        right = right_limit(roots, lane_id+1);
        left  = threads_per_warp()-1-right_limit(__brev(roots), threads_per_warp()-1-lane_id);
        width = rounddown_power_of_2(right - left);
        //printf("%3d: key_mask Ox%08X roots left %d  right %d width %d : %d %d\n", lane_id, roots, left, right, width, __ffs(key_mask), __clz(key_mask));
    }
};

} // namespace impl

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I idx) {
    impl::run_length_type run(idx);

    // get local copies of right and width, which are modified in the reduction loop
    auto rhs = run.right;
    auto width = run.width;

    while (width) {
        unsigned source_lane = run.lane_id + width;

        T source_value = __shfl_sync(run.key_mask, contribution, source_lane);
        if (source_lane<rhs) {
            contribution += source_value;
        }

        rhs = run.left + width;
        width >>= 1;
    }

    if(run.is_root()) {
        // Update atomically in case the run spans multiple warps.
        cuda_atomic_add(target+idx, contribution);
    }
}

} // namespace gpu
} // namespace arb
