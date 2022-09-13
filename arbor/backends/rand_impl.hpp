#pragma once

#include <type_traits>

#include <Random123/boxmuller.hpp>
#include <Random123/threefry.h>

#include <arbor/gpu/gpu_common.hpp>
#include "backends/rand.hpp"

namespace arb {
namespace cbprng {

// checks that chosen random number generator has the correct layout in terms of number of counter
// and key fields
struct traits_ {
    using generator_type = r123::Threefry4x64_R<12>;
    using counter_type = typename generator_type::ctr_type;
    using key_type = counter_type;
    using array_type = counter_type;

    static_assert(counter_type::static_size == cache_size());
    static_assert(std::is_same<typename counter_type::value_type, value_type>::value);
    static_assert(std::is_same<typename generator_type::key_type, key_type>::value);
};

using generator  = traits_::generator_type;
using array_type = traits_::array_type;

struct darray {
    double _data[cache_size()];
    HOST_DEVICE_IF_GPU
    double operator[](unsigned i) const { return _data[i]; }
};

HOST_DEVICE_IF_GPU
inline array_type generate_uniform_random_values(
    value_type seed,
    value_type mech_id,
    value_type var_id,
    value_type gid,
    value_type idx,
    value_type counter) {
    const array_type c{seed, mech_id, var_id, counter};
    const array_type k{gid, idx, 0, 0};
    return generator{}(c, k);
}

HOST_DEVICE_IF_GPU
inline darray generate_normal_random_values(
    value_type seed,
    value_type mech_id,
    value_type var_id,
    value_type gid,
    value_type idx,
    value_type counter) {
    const array_type r = generate_uniform_random_values(seed, mech_id, var_id, gid, idx, counter);
    const auto [n0, n1] = r123::boxmuller(r[0], r[1]);
    const auto [n2, n3] = r123::boxmuller(r[2], r[3]);
    return {n0, n1, n2, n3};
}

} // namespace cbprng
} // namespace arb

