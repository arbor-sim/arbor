#pragma once

#include <cstddef>

#include <arbor/arb_types.hpp>

namespace arb {

namespace cbprng {

using value_type = arb_seed_type;
using counter_type = value_type;

inline constexpr counter_type cache_size() { return 4; }
inline constexpr counter_type cache_index(counter_type c) { return (3u & c); }

} // namespace cbprng

// multicore implementation forward declaration
namespace multicore {
void generate_random_numbers(arb_value_type* dst, std::size_t width, std::size_t width_padded,
    std::size_t num_rv, cbprng::value_type seed, cbprng::value_type mech_id,
    cbprng::value_type counter, arb_size_type const * gid, arb_size_type const * idx);
} // namespace multicore

// gpu implementation forward declaration
namespace gpu {
void generate_random_numbers(arb_value_type* dst, std::size_t width, std::size_t width_padded,
    std::size_t num_rv, cbprng::value_type seed, cbprng::value_type mech_id,
    cbprng::value_type counter, arb_size_type const * gid, arb_size_type const * idx);
} // namespace gpu

} // namespace arb

