#include <array>
#include <Random123/boxmuller.hpp>

#include <arbor/arb_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

#include "backends/rand_impl.hpp"

namespace arb {
namespace gpu {

__global__
void generate_normal_random_values_kernel (
    std::size_t width,
    std::size_t num_variables,
    arb::cbprng::value_type seed, 
    arb::cbprng::value_type mech_id,
    arb::cbprng::value_type counter,
    arb_size_type** prng_indices,
    arb_value_type** dst0,
    arb_value_type** dst1,
    arb_value_type** dst2,
    arb_value_type** dst3
) {
    int const tid = threadIdx.x + blockDim.x*blockIdx.x;
    std::uint64_t const vid = blockIdx.y;

    arb_size_type const* gids = prng_indices[0];
    arb_size_type const* idxs = prng_indices[1];

    if (tid < width) {
        arb::cbprng::value_type const gid = gids[tid];
        arb::cbprng::value_type const idx = idxs[tid];

        const auto r = arb::cbprng::generate_normal_random_values(seed, mech_id, vid, gid, idx, counter);

        dst0[vid][tid] = r[0];
        dst1[vid][tid] = r[1];
        dst2[vid][tid] = r[2];
        dst3[vid][tid] = r[3];
    }
}

void generate_normal_random_values(
    std::size_t width,
    arb::cbprng::value_type seed,
    arb::cbprng::value_type mech_id,
    arb::cbprng::value_type counter,
    memory::device_vector<arb_size_type*>& prng_indices,
    std::array<memory::device_vector<arb_value_type*>, arb::prng_cache_size()>& dst
)
{
    unsigned const block_dim = 128;
    unsigned const grid_dim_x = impl::block_count(width, block_dim);
    unsigned const grid_dim_y = num_variables;

    generate_normal_random_values_kernel<<<dim3{grid_dim_x, grid_dim_y, 1}, block_dim>>>(
        width,
        dst[0].size(),
        seed, 
        mech_id,
        counter,
        prng_indices.data(),
        dst[0].data(), dst[1].data(), dst[2].data(), dst[3].data()
    );
}

} // namespace gpu
} // namespace arb
