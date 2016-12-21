#pragma once

#ifdef NMC_HAVE_CUDA

#include <string>
#include <cstdint>

#include "util.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace nest {
namespace mc {
namespace memory {
namespace gpu {

//
// prototypes for compiled wrappers around fill kernels for GPU memory
//
void fill8(uint8_t* v, uint8_t value, std::size_t n);
void fill16(uint16_t* v, uint16_t value, std::size_t n);
void fill32(uint32_t* v, uint32_t value, std::size_t n);
void fill64(uint64_t* v, uint64_t value, std::size_t n);

//
// helpers for memory where at least on of the target or source is on the gpu
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
} // namespace mc
} // namespace nest

#endif
