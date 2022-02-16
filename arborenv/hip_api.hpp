#include <utility>
#include <string>

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <arborenv/export.hpp>

using DeviceProp = hipDeviceProp_t;

struct ARB_SYMBOL_VISIBLE api_error_type {
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
