#pragma once

#include <sstream>
#include <stdexcept>

#ifdef ARB_CUDA
#include "cuda_api.hpp"
#endif

#ifdef ARB_HIP
#include "hip_api.hpp"
#endif

#define ARB_GPU_CHECK(api_error)                                                     \
    do {                                                                             \
        if (!api_error) {                                                            \
            ::arb::gpu::impl::device_error(api_error, __func__, __FILE__, __LINE__); \
        }                                                                            \
    } while (false)

namespace arb {
namespace gpu {
namespace impl {

inline void device_error(const api_error_type& api_error, const char func[], const char file[], int line) {
    std::ostringstream o;
    o << "device error: \"" << api_error.description() << "\" [" << api_error.name() << "]"
      << " in function: " << func << ", location: " << file << ":" << line;
    throw std::runtime_error(o.str());
}

} // namespace impl
} // namespace gpu
} // namespace arb
