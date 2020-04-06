#include <utility>

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using DeviceProp = cudaDeviceProp;

constexpr auto Success = cudaSuccess;
constexpr auto ErrorInvalidDevice = cudaErrorInvalidDevice;
constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = cudaHostRegisterPortable;

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> cudaError_t {
  return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_string(ARGS&&... args) -> const char* {
    return cudaGetErrorString(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS&&... args) -> cudaError_t {
  return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_memcpy(ARGS&&... args) -> cudaError_t {
    return cudaMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_register(ARGS&&... args) -> cudaError_t {
    return cudaHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_unregister(ARGS&&... args) -> cudaError_t {
    return cudaHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_malloc(ARGS&&... args) -> cudaError_t {
    return cudaMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_free(ARGS&&... args) -> cudaError_t {
    return cudaFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_mem_get_info(ARGS&&... args) -> cudaError_t {
    return cudaMemGetInfo(std::forward<ARGS>(args)...);
}

#else

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
using DeviceProp = hipDeviceProp_t;

constexpr auto Success = hipSuccess;
constexpr auto ErrorInvalidDevice = hipErrorInvalidDevice;
constexpr auto gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = hipHostRegisterPortable;

template <typename... ARGS>
inline auto get_device_properities(ARGS&&... args) -> hipError_t {
    return hipGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_error_string(ARGS&&... args) -> const char* {
    return hipGetErrorString(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto set_device(ARGS&&... args) -> hipError_t {
    return hipSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_memcpy(ARGS&&... args) -> hipError_t {
    return hipMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_register(ARGS&&... args) -> hipError_t {
    return hipHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto host_unregister(ARGS&&... args) -> hipError_t {
    return hipHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_malloc(ARGS&&... args) -> hipError_t {
    return hipMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_free(ARGS&&... args) -> hipError_t {
    return hipFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline auto device_mem_get_info(ARGS&&... args) -> hipError_t {
    return hipMemGetInfo(std::forward<ARGS>(args)...);
}