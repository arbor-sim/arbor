#pragma once
#ifdef ARB_HAVE_HIP 
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#   define HOST_DEVICE_IF_GPU __host__ __device__
#else
#   define HOST_DEVICE_IF_GPU
#endif

namespace arb {
namespace gpu {

// double shuffle for AMD devices
__device__ __inline__ double shfl(double x, int lane)
{
    auto tmp = static_cast<uint64_t>(x);
    auto lo = static_cast<unsigned>(tmp);
    auto hi = static_cast<unsigned>(tmp >> 32);
    hi = __shfl(static_cast<int>(hi), lane, warpSize);
    lo = __shfl(static_cast<int>(lo), lane, warpSize);
    return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                               static_cast<uint64_t>(lo));
}

__device__ __inline__ double gpu_shfl_up(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
#ifdef ARB_HAVE_HIP
    return shfl(idx, lane_id - shift);
#else
    return __shfl_up_sync(key_mask, idx, shift);
#endif
}

__device__ __inline__ double gpu_shfl_down(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
#ifdef ARB_HAVE_HIP 
    return shfl(idx, lane_id + shift);
#else
    return __shfl_up_sync(key_mask, idx, shift);
#endif
}

__device__ __inline__ unsigned gpu_ballot(unsigned mask, unsigned is_root) {
#ifdef ARB_HAVE_HIP
    return __ballot(is_root);
#else
    return __ballot_sync(key_mask, is_root);
#endif
}

__device__ __inline__ unsigned gpu_any(unsigned mask, unsigned width) {
#ifdef ARB_HAVE_HIP
    return __any(width);
#else
    return __any_sync(run.key_mask, width)
#endif
}

namespace impl {
// Number of threads per warp
// This has always been 32, however it may change in future NVIDIA gpus
HOST_DEVICE_IF_GPU
constexpr inline unsigned threads_per_warp() {
#ifdef ARB_HAVE_HIP 
    return 64u;
#else
    return 32u;
#endif
}

// The minimum number of bins required to store n values where the bins have
// dimension of block_size.
HOST_DEVICE_IF_GPU
constexpr inline unsigned block_count(unsigned n, unsigned block_size) {
    return (n+block_size-1)/block_size;
}

} // namespace impl

} // namespace gpu
} // namespace arb
