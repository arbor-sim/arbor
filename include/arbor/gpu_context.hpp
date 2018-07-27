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

struct cuda_device_prop {
#ifdef ARB_GPU_ENABLED
    int cuda_arch;
    cuda_device_prop() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cuda_arch = prop.major*100 + prop.minor;
    }
#endif
};

struct gpu_context {
    bool has_gpu_;
    cuda_device_prop gpu_prop;
    gpu_context(): has_gpu_(false), gpu_prop(){};
    gpu_context(bool has_gpu): has_gpu_(has_gpu), gpu_prop(){};
};

}
