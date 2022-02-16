#include <utility>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <arborenv/export.hpp>

using DeviceProp = cudaDeviceProp;

struct ARB_SYMBOL_VISIBLE api_error_type {
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
