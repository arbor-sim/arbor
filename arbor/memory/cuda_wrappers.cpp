#include <cstdlib>
#include <string>

#include <arbor/arbexcept.hpp>

#include "util.hpp"

#ifdef ARB_HAVE_GPU

#include <cuda.h>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(error, msg)\
throw arbor_exception("CUDA memory:: "+std::string(__func__)+" "+std::string((msg))+": "+cudaGetErrorString(error));

namespace arb {
namespace memory {

using std::to_string;

void cuda_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    if (auto error = cudaMemcpy(dest, src, n, cudaMemcpyDeviceToDevice)) {
        HANDLE_CUDA_ERROR(error, "n="+to_string(n));
    }
}

void cuda_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    if (auto error = cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost)) {
        HANDLE_CUDA_ERROR(error, "n="+to_string(n));
    }
}

void cuda_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    if (auto error = cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice)) {
        HANDLE_CUDA_ERROR(error, "n="+to_string(n));
    }
}

void* cuda_host_register(void* ptr, std::size_t size) {
    if (auto error = cudaHostRegister(ptr, size, cudaHostRegisterPortable)) {
        HANDLE_CUDA_ERROR(error, "unable to register host memory");
    }
    return ptr;
}

void cuda_host_unregister(void* ptr) {
    cudaHostUnregister(ptr);
}

void* cuda_malloc(std::size_t n) {
    void* ptr;

    if (auto error = cudaMalloc(&ptr, n)) {
        HANDLE_CUDA_ERROR(error, "unable to allocate "+to_string(n)+" bytes");
    }
    return ptr;
}

void cuda_free(void* ptr) {
    if (auto error = cudaFree(ptr)) {
        HANDLE_CUDA_ERROR(error, "");
    }
}

} // namespace memory
} // namespace arb

#else

#define NOCUDA \
LOG_ERROR("memory:: "+std::string(__func__)+"(): no CUDA support")

namespace arb {
namespace memory {

void cuda_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void cuda_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void cuda_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    NOCUDA;
}

void* cuda_host_register(void* ptr, std::size_t size) {
    NOCUDA;
    return 0;
}

void cuda_host_unregister(void* ptr) {
    NOCUDA;
}

void* cuda_malloc(std::size_t n) {
    NOCUDA;
    return 0;
}

void cuda_free(void* ptr) {
    NOCUDA;
}

} // namespace memory
} // namespace arb

#endif // def ARB_HAVE_GPU

