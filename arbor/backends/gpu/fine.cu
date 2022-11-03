#include <arbor/fvm_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "fine.hpp"

namespace arb {
namespace gpu {
namespace kernels {
// to[i] = from[p[i]]
template <typename T, typename I>
__global__
void gather(const T* __restrict__ const from,
            T* __restrict__ const to,
            const I* __restrict__ const p,
            unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) {
        to[i] = from[p[i]];
    }
}

// to[p[i]] = from[i]
template <typename T, typename I>
__global__
void scatter(const T* __restrict__ const from,
             T* __restrict__ const to,
             const I* __restrict__ const p,
             unsigned n) {
    unsigned i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i<n) {
        to[p[i]] = from[i];
    }
}

} // namespace kernels

ARB_ARBOR_API void gather(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n)
{
    constexpr unsigned blockdim = 128;
    const unsigned griddim = impl::block_count(n, blockdim);

    kernels::gather<<<griddim, blockdim>>>(from, to, p, n);
}

ARB_ARBOR_API void scatter(
    const arb_value_type* from,
    arb_value_type* to,
    const arb_index_type* p,
    unsigned n)
{
    constexpr unsigned blockdim = 128;
    const unsigned griddim = impl::block_count(n, blockdim);

    kernels::scatter<<<griddim, blockdim>>>(from, to, p, n);
}

} // namespace gpu
} // namespace arb
