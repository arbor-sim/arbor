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
void nrn_mult_(const fvm_index_type* params_, fvm_value_type* state_, int n) {
    int tid_ = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid_<n) {
        state_[tid_] *= params_[tid_];
        //printf("%d %d\n", tid_, params_[tid_]);
    }
}

void nrn_mult(mechanism_ppack_base* p, fvm_value_type* s) {
    auto n = p->width_;
    unsigned block_dim = 128;
    unsigned grid_dim = gpu::impl::block_count(n, block_dim);

    nrn_mult_<<<grid_dim, block_dim>>>(p->coalesced_mult_, s, n);
}

} // namespace gpu
} // namespace arb
