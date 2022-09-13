#pragma once
#include <cstring>
#include <array>

#include "backends/rand.hpp"
#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

void generate_normal_random_values(
    std::size_t width,
    std::size_t num_variables,
    arb::cbprng::value_type seed, 
    arb::cbprng::value_type mech_id,
    arb::cbprng::value_type counter,
    arb_size_type** prng_indices,
    std::array<arb_value_type**, arb::cbprng::cache_size()> dst
);

inline void generate_normal_random_values(
    std::size_t width,
    arb::cbprng::value_type seed, 
    arb::cbprng::value_type mech_id,
    arb::cbprng::value_type counter,
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
