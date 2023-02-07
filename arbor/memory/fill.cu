#include <arbor/gpu/gpu_common.hpp>

#include <cstdint>

namespace arb {
namespace gpu {

template <typename T, typename I>
__global__
void fill_kernel(T* __restrict__ const v, T value, I n) {
    auto tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < n) {
        v[tid] = value;
    }
}

void fill8(uint8_t* v, uint8_t value, std::size_t n) {
    unsigned block_dim = 192;
    launch(impl::block_count(n, block_dim), block_dim, fill_kernel<uint8_t, std::size_t>, v, value, n);
};

void fill16(uint16_t* v, uint16_t value, std::size_t n) {
    unsigned block_dim = 192;
    launch(impl::block_count(n, block_dim), block_dim, fill_kernel<uint16_t, std::size_t>, v, value, n);
};

void fill32(uint32_t* v, uint32_t value, std::size_t n) {
    unsigned block_dim = 192;
    launch(impl::block_count(n, block_dim), block_dim, fill_kernel<uint32_t, std::size_t>, v, value, n);
};

void fill64(uint64_t* v, uint64_t value, std::size_t n) {
    unsigned block_dim = 192;
    launch(impl::block_count(n, block_dim), block_dim, fill_kernel<uint64_t, std::size_t>, v, value, n);
};

} // namespace gpu
} // namespace arb
