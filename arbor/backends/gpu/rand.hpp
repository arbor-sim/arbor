#pragma once
#include <cstring>
#include <array>

#include "../rand.hpp"
#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

void generate_normal_random_values(
    std::size_t   width,
    std::size_t   num_variables,
    cbprng_value_type seed, 
    cbprng_value_type mech_id,
    cbprng_value_type counter,
    arb_size_type** prng_indices,
    std::array<arb_value_type**, cbprng_batch_size> dst
);

inline void generate_normal_random_values(
    std::size_t   width,
    std::uint64_t seed, 
    std::uint64_t mech_id,
    std::uint64_t counter,
    memory::device_vector<arb_size_type*>& prng_indices,
    std::vector<memory::device_vector<arb_value_type*>>& dst
)
{
    generate_normal_random_values(
        width,
        dst[0].size(),
        seed,
        mech_id,
        counter,
        prng_indices.data(),
        {dst[0].data(), dst[1].data(), dst[2].data(), dst[3].data()}
    );
}

} // namespace gpu
} // namespace arb
