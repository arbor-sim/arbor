#pragma once

#include <cstdint>
#include "cuda_atomic.hpp"
#include "cuda_common.hpp"

namespace arb {
namespace gpu {

namespace impl{

constexpr unsigned mask_all = 0xFFFFFFFF;

// return the first power of 2 that is less than or equal to i
__device__ __inline__
unsigned rounddown_power_of_2(std::uint32_t i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(31u - __clz(i));
}

// Return the first power of 2 that is larger than or equal to i
// input i is in the closed interval [0, 2^31]
// The result for i>2^31 is invalid, because it can't be stored in an
// unsigned 32 bit integer.
// This isn't a problem for the use case that we will use it for, because
// it will be used for values in the range [0, threads_per_warp]
__device__ __inline__
unsigned roundup_power_of_2(std::uint32_t i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(32u - __clz(i));
}

// run_length Stores information about a run length.
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
    unsigned left;
    unsigned right;
    unsigned shift;
    unsigned lane_id;
    unsigned key_mask;

    __device__ __inline__
    bool is_root() const {
        return left == lane_id;
    }

    __device__
    run_length(int idx) {
        lane_id = threadIdx.x%threads_per_warp();
        __syncwarp();
        // TODO: calculate key mask directly from array sizes (outside main loop)
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

        // determine the bounds of the lanes with the same key as idx
        right = right_limit(roots, lane_id+1);
        left  = threads_per_warp()-1-right_limit(__brev(roots), threads_per_warp()-1-lane_id);

        // find the largest power of two that is less than or equal to the run length
        shift = rounddown_power_of_2(right - left);
    }
};

} // namespace impl

template <typename T, typename I>
__device__ __inline__
void reduce_by_key(T contribution, T* target, I i) {
    impl::run_length run(i);

    unsigned shift = run.shift;
    const unsigned key_lane = run.lane_id - run.left;

    bool participate = run.lane_id+shift<run.right;

    while (__any_sync(run.key_mask, shift)) {
        const unsigned w = participate? shift: 0;
        const unsigned source_lane = run.lane_id + w;

        T source_value = __shfl_sync(run.key_mask, contribution, source_lane);
        if (participate) {
            contribution += source_value;
        }

        shift >>= 1;
        participate = key_lane<shift;
    }

    if(run.is_root()) {
        // The update must be atomic, because the run may span multiple warps.
        cuda_atomic_add(target+i, contribution);
    }
}

} // namespace gpu
} // namespace arb
