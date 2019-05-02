#include <iostream>
#include <backends/event.hpp>
#include <backends/multi_event_stream_state.hpp>
#include <backends/gpu/cuda_common.hpp>
#include <backends/gpu/math_cu.hpp>
#include <backends/gpu/mechanism_ppack_base.hpp>
#include <backends/gpu/reduce_by_key.hpp>

namespace arb {
namespace gpu {

__global__
void multiply_in_place_(fvm_value_type* s, const fvm_index_type* p, int n) {
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid_<n) {
        s[tid_] *= p[tid_];
    }
}

void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n) {
    unsigned block_dim = 128;
    unsigned grid_dim = gpu::impl::block_count(n, block_dim);

    multiply_in_place_<<<grid_dim, block_dim>>>(s, p, n);
}

} // namespace gpu
} // namespace arb
