#pragma once

#include <sstream>
#include <stdexcept>

#include "gpu_api.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#   define HOST_DEVICE_IF_GPU __host__ __device__
#else
#   define HOST_DEVICE_IF_GPU
#endif

namespace arb {
namespace gpu {

namespace impl {
// Number of threads per warp
// This has always been 32, however it may change in future NVIDIA gpus
HOST_DEVICE_IF_GPU
constexpr inline unsigned threads_per_warp() {
#ifdef ARB_HIP
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

inline void device_error(const api_error_type& api_error, const char func[], const char file[], int line) {
    std::ostringstream o;
    o << "device error: \"" << api_error.description() << "\" [" << api_error.name() << "]"
      << " in function: " << func << ", location: " << file << ":" << line;
    throw std::runtime_error(o.str());
}

} // namespace impl

#define ARB_GPU_CHECK(api_error)                          \
    do {                                                  \
        if (!api_error) {                                 \
            ::arb::gpu::impl::device_error(               \
                api_error, __func__, __FILE__, __LINE__); \
        }                                                 \
    } while (false)

template<typename Kernel, typename... Args>
void launch(const dim3& blocks, const dim3& threads, Kernel kernel, Args&&... args) {
    kernel<<<blocks, threads>>>(std::forward<Args>(args)...);
    ARB_GPU_CHECK(get_last_error());
#ifndef NDEBUG
    ARB_GPU_CHECK(device_synchronize());
#endif
}

template<typename Kernel, typename... Args>
void launch_1d(unsigned elements, unsigned block_size, Kernel kernel, Args&&... args) {
    launch(impl::block_count(elements, block_size), block_size, kernel, std::forward<Args>(args)...);
}

} // namespace gpu
} // namespace arb
