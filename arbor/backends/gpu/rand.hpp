#pragma once
#include <cstring>
#include <array>

#include "backends/rand.hpp"
#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

// dispatches to cuda kernel
void generate_normal_random_values(
    std::size_t width,                                        // number of sites
    arb::cbprng::value_type seed,                             // simulation seed value
    arb::cbprng::value_type mech_id,                          // mechanism id
    arb::cbprng::value_type counter,                          // step counter
    memory::device_vector<arb_size_type*>& prng_indices,      // holds the gid and per-cell location indices
    std::array<memory::device_vector<arb_value_type*>, arb::prng_cache_size()>& dst  // pointers to random number cache
);

} // namespace gpu
} // namespace arb
