#pragma once

namespace arb {
namespace memory {

void gpu_memcpy_d2d(void* dest, const void* src, std::size_t n);
void gpu_memcpy_d2h(void* dest, const void* src, std::size_t n);
void gpu_memcpy_h2d(void* dest, const void* src, std::size_t n);
void* gpu_host_register(void* ptr, std::size_t size);
void gpu_host_unregister(void* ptr);
void* gpu_malloc(std::size_t n);
void gpu_free(void* ptr);

} // namespace memory
} // namespace arb
