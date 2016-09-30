#include <cstdlib>
#include <cstdint>

namespace memory {
namespace gpu {
    template <typename T, typename I>
    __global__
    void fill_kernel(T* v, T value, I n) {
        std::size_t tid = threadIdx.x + blockDim.x*blockIdx.x;
        std::size_t grid_step = blockDim.x * gridDim.x;

        while(tid < n) {
            v[tid] = value;
            tid += grid_step;
        }
    }

    void fill8(uint8_t* v, uint8_t value, std::size_t n) {
        auto thread_dim = int{192};
        dim3 dim_block(thread_dim);
        dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0));

        fill_kernel<uint8_t><<<dim_grid, dim_block>>>(v, value, n);
    };

    void fill16(uint16_t* v, uint16_t value, std::size_t n) {
        auto thread_dim = int{192};
        dim3 dim_block(thread_dim);
        dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0));

        fill_kernel<uint16_t><<<dim_grid, dim_block>>>(v, value, n);
    };

    void fill32(uint32_t* v, uint32_t value, std::size_t n) {
        auto thread_dim = int{192};
        dim3 dim_block(thread_dim);
        dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0));

        fill_kernel<uint32_t><<<dim_grid, dim_block>>>(v, value, n);
    };

    void fill64(uint64_t* v, uint64_t value, std::size_t n) {
        auto thread_dim = int{192};
        dim3 dim_block(thread_dim);
        dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0));

        fill_kernel<uint64_t><<<dim_grid, dim_block>>>(v, value, n);
    };
} // namespace gpu
} // namespace memory

