#include <utility>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <arbor/export.hpp>

namespace arb {
namespace gpu {

/// Device queries

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

    std::string name() const {
        std::string s = cudaGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = cudaGetErrorString(value);
        return s;
    }
};

inline api_error_type get_last_error() {
    return cudaGetLastError();
}

inline api_error_type device_synchronize() {
    return cudaDeviceSynchronize();
}

constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = cudaHostRegisterPortable;

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS &&... args) {
    return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type set_device(ARGS &&... args) {
    return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy(ARGS &&... args) {
    return cudaMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy_async(ARGS &&... args) {
    return cudaMemcpyAsync(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_register(ARGS &&... args) {
    return cudaHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_unregister(ARGS &&... args) {
    return cudaHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_malloc(ARGS &&... args) {
    return cudaMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_free(ARGS &&... args) {
    return cudaFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_mem_get_info(ARGS &&... args) {
    return cudaMemGetInfo(std::forward<ARGS>(args)...);
}

#ifdef __CUDACC__
/// Atomics

// Wrappers around CUDA addition functions.
// CUDA 8 introduced support for atomicAdd with double precision, but only for
// Pascal GPUs (__CUDA_ARCH__ >= 600). We assume no one is running anything older.
__device__
inline double gpu_atomic_add(double* address, double val) {
    return atomicAdd(address, val);
}

__device__
inline double gpu_atomic_sub(double* address, double val) {
    return gpu_atomic_add(address, -val);
}

__device__
inline float gpu_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float gpu_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

/// Warp-Level Primitives
__device__ __inline__ unsigned ballot(unsigned mask, unsigned is_root) {
    return __ballot_sync(mask, is_root);
}

__device__ __inline__ unsigned active_mask() {
    return __activemask();
}

__device__ __inline__ unsigned any(unsigned mask, unsigned width) {
    return __any_sync(mask, width);
}

template<typename T>
__device__ __inline__ T shfl_up(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return __shfl_up_sync(mask, var, shift);
}

template<typename T>
__device__ __inline__ T shfl_down(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return __shfl_down_sync(mask, var, shift);
}
#endif

} // namespace gpu
} // namespace arb
