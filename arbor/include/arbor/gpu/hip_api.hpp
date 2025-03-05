#include <utility>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <arbor/export.hpp>

namespace arb {
namespace gpu {

/// Device queries

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

    std::string name() const {
        std::string s = hipGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = hipGetErrorString(value);
        return s;
    }
};

inline api_error_type get_last_error() {
    return hipGetLastError();
}

inline api_error_type device_synchronize() {
    return hipDeviceSynchronize();
}

constexpr auto gpuMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = hipHostRegisterPortable;

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS&&... args) {
    return hipGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type set_device(ARGS&&... args) {
    return hipSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy(ARGS&&... args) {
    return hipMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy_async(ARGS &&... args) {
    return hipMemcpyAsync(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_register(ARGS&&... args) {
    return hipHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_unregister(ARGS&&... args) {
    return hipHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_malloc(ARGS&&... args) {
    return hipMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_free(ARGS&&... args) {
    return hipFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_mem_get_info(ARGS&&... args) {
    return hipMemGetInfo(std::forward<ARGS>(args)...);
}

/// Atomics

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

/// Warp-level Primitives

template<typename T>
__device__ __inline__
std::enable_if_t< !std::is_same_v<std::decay_t<T>, double>, std::decay_t<T>>
shfl(T x, int lane) {
    return __shfl(x, lane);
}

__device__ __inline__ double shfl(double x, int lane)
{
    auto tmp = static_cast<uint64_t>(x);
    auto lo = static_cast<unsigned>(tmp);
    auto hi = static_cast<unsigned>(tmp >> 32);
    hi = __shfl(static_cast<int>(hi), lane, warpSize);
    lo = __shfl(static_cast<int>(lo), lane, warpSize);
    return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                               static_cast<uint64_t>(lo));
}

__device__ __inline__ unsigned ballot(unsigned mask, unsigned is_root) {
    return __ballot(is_root);
}

__device__ __inline__ unsigned active_mask() {
    return __activemask();
}

__device__ __inline__ unsigned any(unsigned mask, unsigned width) {
    return __any(width);
}

template<typename T>
__device__ __inline__ T shfl_up(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return shfl(var, (int)lane_id - shift);
}

template<typename T>
__device__ __inline__ T shfl_down(unsigned mask, T var, unsigned lane_id, unsigned shift) {
    return shfl(var, (int)lane_id + shift);
}

} // namespace gpu
} // namespace arb

