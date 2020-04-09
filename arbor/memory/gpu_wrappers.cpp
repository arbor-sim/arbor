#include <cstdlib>
#include <string>

#include <arbor/arbexcept.hpp>

#include "util.hpp"

#ifdef ARB_HAVE_GPU

#include <backends/gpu/gpu_api.hpp>

#define HANDLE_GPU_ERROR(error, msg)\
throw arbor_exception("GPU memory:: "+std::string(__func__)+" "+std::string((msg))+": "+device_error_string(error));

namespace arb {
namespace memory {

using std::to_string;
using namespace gpu;
 
void gpu_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    if (auto error = device_memcpy(dest, src, n, gpuMemcpyDeviceToDevice)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void gpu_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    if (auto error = device_memcpy(dest, src, n, gpuMemcpyDeviceToHost)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void gpu_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    if (auto error = device_memcpy(dest, src, n, gpuMemcpyHostToDevice)) {
        HANDLE_GPU_ERROR(error, "n="+to_string(n));
    }
}

void* gpu_host_register(void* ptr, std::size_t size) {
    if (auto error = host_register(ptr, size, gpuHostRegisterPortable)) {
        HANDLE_GPU_ERROR(error, "unable to register host memory");
    }
    return ptr;
}

void gpu_host_unregister(void* ptr) {
    host_unregister(ptr);
}

void* gpu_malloc(std::size_t n) {
    void* ptr;

    if (auto error = device_malloc(&ptr, n)) {
        HANDLE_GPU_ERROR(error, "unable to allocate "+to_string(n)+" bytes");
    }
    return ptr;
}

void gpu_free(void* ptr) {
    if (auto error = device_free(ptr)) {
        HANDLE_GPU_ERROR(error, "");
    }
}

} // namespace memory
} // namespace arb

#else

#define NOGPU \
LOG_ERROR("memory:: "+std::string(__func__)+"(): no GPU support")

namespace arb {
namespace memory {

void gpu_memcpy_d2d(void* dest, const void* src, std::size_t n) {
    NOGPU;
}

void gpu_memcpy_d2h(void* dest, const void* src, std::size_t n) {
    NOGPU;
}

void gpu_memcpy_h2d(void* dest, const void* src, std::size_t n) {
    NOGPU;
}

void* gpu_host_register(void* ptr, std::size_t size) {
    NOGPU;
    return 0;
}

void gpu_host_unregister(void* ptr) {
    NOGPU;
}

void* gpu_malloc(std::size_t n) {
    NOGPU;
    return 0;
}

void gpu_free(void* ptr) {
    NOGPU;
}

} // namespace memory
} // namespace arb

#endif // def ARB_HAVE_GPU

