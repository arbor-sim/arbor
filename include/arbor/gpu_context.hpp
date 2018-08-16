#pragma once

#include <memory>
#include <string>
#include <iostream>
#include <arbor/version.hpp>

#ifdef ARB_GPU_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace arb {

#ifndef ARB_GPU_ENABLED
struct gpu_context {
    bool has_gpu_ = false;
    size_t attributes = 0;
};
#else
enum gpu_flags {
    no_sync = 0,
    has_atomic_double = 1
};

struct gpu_context {
    bool has_gpu_ = true;
    size_t attributes = 0;
    gpu_context(): attributes(get_attributes()) {};

private:
    size_t get_attributes() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        if(prop.concurrentManagedAccess)
            attributes |= (1 << gpu_flags::no_sync);
        if(prop.major*100 + prop.minor >= 600)
            attributes |= (1 << gpu_flags::has_atomic_double);
        return attributes;
    };
};
#endif

}
