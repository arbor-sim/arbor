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
    has_concurrent_managed_access = 1,
    has_atomic_double = 2
};

struct gpu_context {
    bool has_gpu_;
    size_t attributes_;

    gpu_context() : has_gpu_(true) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        attributes_ = 0;
        if (prop.concurrentManagedAccess) {
            attributes_ |= gpu_flags::has_concurrent_managed_access;
        }
        if (prop.major*100 + prop.minor >= 600) {
            attributes_ |= gpu_flags::has_atomic_double;
        }
    };

    bool has_concurrent_managed_access() {
        return attributes_ & gpu_flags::has_concurrent_managed_access;
    }

    bool has_atomic_double() {
        return attributes_ & gpu_flags::has_atomic_double;
    }

    void synchronize_for_managed_access() {
        if(!has_concurrent_managed_access()) {
            cudaDeviceSynchronize();
        }
    }
};

#endif
}
