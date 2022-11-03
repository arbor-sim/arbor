#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "backends/rand_impl.hpp"

namespace arb {
namespace gpu {

namespace kernel {
__global__
void generate_random_numbers(
    arb_value_type* __restrict__ dst,
    std::size_t width,
    std::size_t width_padded,
    std::size_t num_rv,
    arb::cbprng::value_type seed,
    arb::cbprng::value_type mech_id,
    arb::cbprng::value_type counter,
    arb_size_type const * __restrict__ gids,
    arb_size_type const * __restrict__ idxs,
    unsigned cache_size) {
    // location and variable number extracted from thread block
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    const arb::cbprng::value_type n = blockIdx.y;

    if (i < width) {
        const arb::cbprng::value_type gid = gids[i];
        const arb::cbprng::value_type idx = idxs[i];
        const auto r = arb::cbprng::generator{}(arb::cbprng::array_type{seed, mech_id, n, counter},
            arb::cbprng::array_type{gid, idx, 0xdeadf00dull, 0xdeadbeefull});
        const auto a = r123::boxmuller(r[0], r[1]);
        const auto b = r123::boxmuller(r[2], r[3]);
        dst[i + width_padded*(0 + cache_size*n)] = a.x;
        dst[i + width_padded*(1 + cache_size*n)] = a.y;
        dst[i + width_padded*(2 + cache_size*n)] = b.x;
        dst[i + width_padded*(3 + cache_size*n)] = b.y;
    }
}
} // namespace kernel

void generate_random_numbers(
    arb_value_type* dst,        // points to random number storage
    std::size_t width,          // number of sites
    std::size_t width_padded,   // padded number of sites
    std::size_t num_rv,         // number of random variables
    cbprng::value_type seed,    // simulation seed value
    cbprng::value_type mech_id, // mechanism id
    cbprng::value_type counter, // step counter
    arb_size_type const * gid,  // global cell ids (size = width)
    arb_size_type const * idx   // per-cell location index (size = width)
    ) {
    using impl::block_count;

    unsigned const block_dim = 128;
    unsigned const grid_dim_x = block_count(width, block_dim);
    unsigned const grid_dim_y = num_rv;

    kernel::generate_random_numbers<<<dim3{grid_dim_x, grid_dim_y, 1}, block_dim>>>(
        dst, width, width_padded, num_rv, seed, mech_id, counter, gid, idx, cbprng::cache_size());
}

} // namespace gpu
} // namespace arb

