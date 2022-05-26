#pragma once

#include <cstdint>

namespace arb {
namespace math {

double normal_rand(
    std::uint64_t i_,
    std::uint64_t seed,
    std::uint64_t gid,
    double time,
    std::uint64_t mech_id,
    std::uint64_t mech_inst,
    std::uint64_t var_id);

} // namespace math
} // namespace arb
