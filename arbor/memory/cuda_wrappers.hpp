#pragma once

namespace arb {
namespace memory {

void cuda_memcpy_d2d(void* dest, const void* src, std::size_t n);
void cuda_memcpy_d2h(void* dest, const void* src, std::size_t n);
void cuda_memcpy_h2d(void* dest, const void* src, std::size_t n);
void* cuda_host_register(void* ptr, std::size_t size);
void cuda_host_unregister(void* ptr);
void* cuda_malloc(std::size_t n);
void cuda_free(void* ptr);

} // namespace memory
} // namespace arb
