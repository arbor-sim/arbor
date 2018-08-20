#include <memory>

#ifdef ARB_HAVE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace arb {

#ifndef ARB_HAVE_GPU
struct gpu_context {
    bool has_gpu_;
    size_t attributes_;

    gpu_context(): has_gpu_(false), attributes_(0) {}
};

#else

enum gpu_flags {
    has_concurrent_managed_access = 0,
    has_atomic_double = 1
};

struct gpu_context {
    bool has_gpu_;
    size_t attributes_;

    gpu_context() : has_gpu_(true) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        attributes_ = 0; 
        if (prop.concurrentManagedAccess) {
            attributes_ |= (1 << gpu_flags::has_concurrent_managed_access);
        }
        if (prop.major*100 + prop.minor >= 600) {
            attributes_ |= (1 << gpu_flags::has_atomic_double);
        }
    };

    bool has_concurrent_managed_access() {
        return attributes_ & 1 << gpu_flags::has_concurrent_managed_access;
    }

    bool has_atomic_double() {
        return attributes_ & 1 << gpu_flags::has_atomic_double;
    }

    void synchronize() {
        if(has_concurrent_managed_access()) {
            cudaDeviceSynchronize();
        }
    }
};

#endif
}
