#include <utility>
#include <string>

#ifdef ARB_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using DeviceProp = cudaDeviceProp;

struct api_error_type {
    cudaError_t value;
    api_error_type(cudaError_t e): value(e) {}

    operator bool() const {
        return value==cudaSuccess;
    }

    bool is_invalid_device() const {
        return value == cudaErrorInvalidDevice;
    }

    bool no_device_found() const {
        return value == cudaErrorNoDevice;
    }

    std::string name() const {
        std::string s = cudaGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = cudaGetErrorString(value);
        return s;
    }
};

template <typename... ARGS>
inline api_error_type get_device_count(ARGS&&... args) {
    return cudaGetDeviceCount(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS&&... args) {
    return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}
#endif

#ifdef ARB_HIP
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using DeviceProp = hipDeviceProp_t;

struct api_error_type {
    hipError_t value;
    api_error_type(hipError_t e): value(e) {}

    operator bool() const {
        return value==hipSuccess;
    }

    bool is_invalid_device() const {
        return value == hipErrorInvalidDevice;
    }

    bool no_device_found() const {
        return value == hipErrorNoDevice;
    }

    std::string name() const {
        std::string s = hipGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = hipGetErrorString(value);
        return s;
    }
};

template <typename... ARGS>
inline api_error_type get_device_count(ARGS&&... args) {
    return hipGetDeviceCount(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS&&... args) {
    return hipGetDeviceProperties(std::forward<ARGS>(args)...);
}
#endif
