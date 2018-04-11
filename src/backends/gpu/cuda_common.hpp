#pragma once

#ifdef __CUDACC__
#   define HOST_DEVICE_IF_CUDA __host__ __device__
#else
#   define HOST_DEVICE_IF_CUDA
#endif

namespace arb {
namespace gpu {

namespace impl {
// Number of threads per warp
// This has always been 32, however it may change in future NVIDIA gpus
HOST_DEVICE_IF_CUDA
constexpr inline unsigned threads_per_warp() {
    return 32u;
}

// The minimum number of bins required to store n values where the bins have
// dimension of block_size.
HOST_DEVICE_IF_CUDA
constexpr inline unsigned block_count(unsigned n, unsigned block_size) {
    return (n+block_size-1)/block_size;
}

} // namespace impl

} // namespace gpu
} // namespace arb
