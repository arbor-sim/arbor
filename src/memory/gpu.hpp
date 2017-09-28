#pragma once

#ifdef ARB_HAVE_GPU

#include <string>
#include <cstdint>

#include "util.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace arb {
namespace memory {
namespace gpu {

//
// helpers for memory where at least one of the target or source is on the gpu
//
template <typename T>
void memcpy_d2h(const T* from, T* to, std::size_t size) {
    auto bytes = sizeof(T)*size;
    if (size==0) return;
    auto status = cudaMemcpy(
        reinterpret_cast<void*>(to), reinterpret_cast<const void*>(from),
        bytes, cudaMemcpyDeviceToHost
    );
    if(status != cudaSuccess) {
        LOG_ERROR("cudaMemcpy(d2h, " + std::to_string(bytes) + ") " + cudaGetErrorString(status));
        abort();
    }
}

template <typename T>
void memcpy_h2d(const T* from, T* to, std::size_t size) {
    auto bytes = sizeof(T)*size;
    if (size==0) return;
    auto status = cudaMemcpy(
        reinterpret_cast<void*>(to), reinterpret_cast<const void*>(from),
        bytes, cudaMemcpyHostToDevice
    );
    if(status != cudaSuccess) {
        LOG_ERROR("cudaMemcpy(h2d, " + std::to_string(bytes) + ") " + cudaGetErrorString(status));
        abort();
    }
}

template <typename T>
void memcpy_d2d(const T* from, T* to, std::size_t size) {
    auto bytes = sizeof(T)*size;
    if (size==0) return;
    auto status = cudaMemcpy(
        reinterpret_cast<void*>(to), reinterpret_cast<const void*>(from),
        bytes, cudaMemcpyDeviceToDevice
    );
    if(status != cudaSuccess) {
        LOG_ERROR("cudaMemcpy(d2d, " + std::to_string(bytes) + ") " + cudaGetErrorString(status));
        abort();
    }
}

} // namespace gpu
} // namespace memory
} // namespace arb

#endif
