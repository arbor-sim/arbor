#include <utility>

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using DeviceProp = cudaDeviceProp;
using Error = cudaError_t;

constexpr auto Success = cudaSuccess;
constexpr auto ErrorInvalidDevice = cudaErrorInvalidDevice;
constexpr auto ErrorNoDevice = cudaErrorNoDevice;

template <typename... ARGS>
inline auto get_device_count(ARGS&&... args) -> StatusType {
    return CudaGetDeviceCount(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> cudaError_t {
    return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_string(ARGS&&... args) -> const char* {
    return cudaGetErrorString(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_name(ARGS&&... args) -> const char* {
    return cudaGetErrorName(std::forward<ARGS>(args)...);
}
#else

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

using DeviceProp = hipDeviceProp_t;
using Error = hipError_t;

constexpr auto Success = hipSuccess;
constexpr auto ErrorInvalidDevice = hipErrorInvalidDevice;
constexpr auto ErrorNoDevice = hipErrorNoDevice;

template <typename... ARGS>
inline auto get_device_count(ARGS&&... args) -> hipError_t {
    return hipGetDeviceCount(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> hipError_t {
    return hipGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_string(ARGS&&... args) -> const char* {
    return hipGetErrorString(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_name(ARGS&&... args) -> const char* {
    return hipGetErrorName(std::forward<ARGS>(args)...);
}
