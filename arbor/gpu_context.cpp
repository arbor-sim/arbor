#include <memory>

#include <arbor/gpu_context.hpp>

#ifdef ARB_HAVE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace arb {

#ifndef ARB_HAVE_GPU

gpu_context::gpu_context(): has_gpu_(false), attributes_(0) {}

#else

enum gpu_flags {
    has_concurrent_managed_access = 0,
    has_atomic_double = 1
};

size_t get_attributes() {
    size_t attributes = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(prop.concurrentManagedAccess)
        attributes |= (1 << gpu_flags::has_concurrent_managed_access);
    if(prop.major*100 + prop.minor >= 600)
        attributes |= (1 << gpu_flags::has_atomic_double);
    return attributes;
};

gpu_context::gpu_context(): has_gpu_(true), attributes_(get_attributes()) {};

bool gpu_context::has_concurrent_managed_access() {
    return attributes_ & 1 << gpu_flags::has_concurrent_managed_access;
}

bool gpu_context::has_atomic_double() {
    return attributes_ & 1 << gpu_flags::has_atomic_double;
}

#endif

}
