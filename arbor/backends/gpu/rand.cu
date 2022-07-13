#include <array>
#include <Random123/boxmuller.hpp>
#include <Random123/threefry.h>

#include <arbor/arb_types.hpp>
#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

namespace arb {
namespace gpu {

__global__
void generate_normal_random_values_kernel (
    std::size_t   width,
    std::size_t   num_variables,
    std::uint64_t seed, 
    std::uint64_t mech_id,
    std::uint64_t counter,
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
        std::uint64_t const gid = gids[tid];
        std::uint64_t const idx = idxs[tid];
        using rng = r123::Threefry4x64_R<12>;
        using counter_type = typename rng::key_type;
        using key_type = typename rng::key_type;

        static_assert(std::is_same<typename counter_type::value_type, std::uint64_t>::value,
            "64 bit width");
        static_assert(std::is_same<typename key_type::value_type, std::uint64_t>::value,
            "64 bit width");
        static_assert(counter_type::static_size == 4, "size of array");
        static_assert(key_type::static_size == 4, "size of array");

        counter_type c{seed, mech_id, vid, counter};
        key_type k{gid, idx, 0, 0};

        const auto r = rng{}(c, k);
        const auto n0 = r123::boxmuller(r[0], r[1]);
        const auto n1 = r123::boxmuller(r[2], r[3]);

        dst0[vid][tid] = n0.x;
        dst1[vid][tid] = n0.y;
        dst2[vid][tid] = n1.x;
        dst3[vid][tid] = n1.y;
    }
}
 

void generate_normal_random_values(
    std::size_t   width,
    std::size_t   num_variables,
    std::uint64_t seed, 
    std::uint64_t mech_id,
    std::uint64_t counter,
    arb_size_type** prng_indices,
    std::array<arb_value_type**, 4> dst
)
{
    unsigned const block_dim = 128;
    unsigned const grid_dim_x = impl::block_count(width, block_dim);
    unsigned const grid_dim_y = num_variables;

    generate_normal_random_values_kernel<<<dim3{grid_dim_x, grid_dim_y, 1}, block_dim>>>(
    width,
    num_variables,
    seed, 
    mech_id,
    counter,
    prng_indices,
    dst[0], dst[1], dst[2], dst[3]
    );
}

} // namespace gpu
} // namespace arb
