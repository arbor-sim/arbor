#pragma once
#include <cstring>
#include <array>
#include <Random123/boxmuller.hpp>

#include "../rand.hpp"

namespace arb {
namespace multicore {

std::array<arb_value_type,cbprng_batch_size> generate_normal_random_values(
    cbprng_value_type seed,
    cbprng_value_type mech_id,
    cbprng_value_type var_id,
    cbprng_value_type gid,
    cbprng_value_type idx,
    cbprng_value_type counter) {

    using counter_type = typename cbprng_generator::ctr_type;
    using key_type = typename cbprng_generator::key_type;

    counter_type c{seed, mech_id, var_id, counter};
    key_type k{gid, idx, 0, 0};

    const auto r = cbprng_generator{}(c, k);
    const auto n0 = r123::boxmuller(r[0], r[1]);
    const auto n1 = r123::boxmuller(r[2], r[3]);
    return {n0.x, n0.y, n1.x, n1.y};
}

} // namespace multicore
} // namespace arb

