#pragma once

#include <cstdint>
#include "detail.hpp"
#include <backends/gpu/intrinsics.hpp>

namespace arb {
namespace gpu {

namespace impl{

constexpr unsigned mask_all = 0xFFFFFFFF;



// Wrappers around the CUDA warp intrinsics used in this file.
// CUDA 9 replaced the warp intrinsics with _sync variants, and
// depricated the old symbols.
// These wrappers can be removed CUDA 9 becomnse the minimum
// version used by Arbor.

template <typename T>
static __device__ __inline__
T arb_shfl(T x, unsigned lane) {
#if __CUDACC_VER_MAJOR__ < 9
    return __shfl(x, lane);
#else
    return __shfl_sync(mask_all, x, lane);
#endif
}

template <typename T>
static __device__ __inline__
unsigned arb_shfl_up(T x, unsigned delta) {
#if __CUDACC_VER_MAJOR__ < 9
    return __shfl_up(x, delta);
#else
    return __shfl_up_sync(mask_all, x, delta);
#endif
}

static __device__ __inline__
unsigned arb_ballot(int pred) {
#if __CUDACC_VER_MAJOR__ < 9
    return __ballot(pred);
#else
    return __ballot_sync(mask_all, pred);
#endif
}

// return the power of 2 that is less than or equal to i
__device__ __inline__
unsigned rounddown_power_of_2(std::uint32_t i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(31u - __clz(i));
}

// The __shfl warp intrinsic is only implemented for 32 bit values.
// To shuffle a 64 bit value (usually a double), the value must be copied
// with two 32 bit shuffles.
// get_from_lane is a wrapper around __shfl() for both single and double
// precision.
// TODO: CUDA 9 provides a double precision version of __shfl that
// implements this hack. When we make CUDA 9 the minimum version, use
// the CUDA implementation instead.

__device__ __inline__
double get_from_lane(double x, unsigned lane) {
    // split the double number into two 32b registers.
    int lo, hi;

    asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

    // shuffle the two 32b registers.
    lo = arb_shfl(lo, lane);
    hi = arb_shfl(hi, lane);

    // return the recombined 64 bit value
    return __hiloint2double(hi, lo);
}

__device__ __inline__
float get_from_lane(float value, unsigned lane) {
    return arb_shfl(value, lane);
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
        auto right_limit = [] (unsigned roots, unsigned shift) {
            unsigned zeros_right  = __ffs(roots>>shift);
            return zeros_right ? shift -1 + zeros_right : threads_per_warp();
        };

        lane_id = threadIdx.x%threads_per_warp();

        // determine if this thread is the root
        int left_idx  = arb_shfl_up(idx, 1);
        int is_root = 1;
        if(lane_id>0) {
            is_root = (left_idx != idx);
        }

        // determine the range this thread contributes to
        unsigned roots = arb_ballot(is_root);

        right = right_limit(roots, lane_id+1);
        left  = threads_per_warp()-1-right_limit(__brev(roots), threads_per_warp()-1-lane_id);
        width = rounddown_power_of_2(right - left);
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

        auto source_value = impl::get_from_lane(contribution, source_lane);
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
