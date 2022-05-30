// GPU kernels and wrappers for shared state methods.

#include <cstdint>

#include <backends/event.hpp>
#include <backends/event_stream_state.hpp>

#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

namespace arb {
namespace gpu {

namespace kernel {

// Vector/scalar addition: x[i] += v ∀i
template <typename T>
__global__ void add_scalar(unsigned n,
                           T* __restrict__ const x,
                           fvm_value_type v) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] += v;
    }
}

__global__ void take_samples_impl(
    event_stream_state<raw_probe_info> s,
    const fvm_value_type time,
    fvm_value_type* __restrict__ const sample_time,
    fvm_value_type* __restrict__ const sample_value)
{
    const unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    const auto begin = s.begin_marked;
    const auto end = s.end_marked;
    const unsigned nsamples = end - begin;
    if (i<nsamples) {
        auto p = begin+i;
        sample_time[p->offset] = time;
        sample_value[p->offset] = p->handle? *p->handle: 0;
    }
}

} // namespace kernel

using impl::block_count;

void add_scalar(std::size_t n, fvm_value_type* data, fvm_value_type v) {
    if (!n) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(n, block_dim);
    kernel::add_scalar<<<nblock, block_dim>>>(n, data, v);
}

void take_samples_impl(
    const event_stream_state<raw_probe_info>& s,
    const fvm_value_type& time, fvm_value_type* sample_time, fvm_value_type* sample_value)
{
    constexpr int block_dim = 128;
    const int nsamples = s.end_marked - s.begin_marked;
    if (nsamples) {
        const int nblock = block_count(nsamples, block_dim);
        kernel::take_samples_impl<<<nblock, block_dim>>>(s, time, sample_time, sample_value);
    }
}

} // namespace gpu
} // namespace arb
