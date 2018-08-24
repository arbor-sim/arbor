#include <memory>

#ifdef ARB_HAVE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "gpu_context.hpp"

namespace arb {

bool gpu_context::has_concurrent_managed_access() const {
    return attributes_ & gpu_flags::has_concurrent_managed_access;
}

bool gpu_context::has_atomic_double() const {
    return attributes_ & gpu_flags::has_atomic_double;
}

#ifndef ARB_HAVE_GPU

gpu_context::gpu_context(): has_gpu_(false), attributes_(0) {}
void gpu_context::synchronize_for_managed_access() const {}

#else

gpu_context::gpu_context(): has_gpu_(true), attributes_(0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.concurrentManagedAccess) {
        attributes_ |= gpu_flags::has_concurrent_managed_access;
    }
    if (prop.major*100 + prop.minor >= 600) {
        attributes_ |= gpu_flags::has_atomic_double;
    }
};

void gpu_context::synchronize_for_managed_access() const {
    if(!has_concurrent_managed_access()) {
        cudaDeviceSynchronize();
    }
}

#endif

std::shared_ptr<gpu_context> make_gpu_context() {
    return std::make_shared<gpu_context>();
}

}
