#pragma once

#include <arbor/arb_types.hpp>

namespace arb {
namespace cbprng {

using value_type = arb_seed_type;
using counter_type = value_type;

inline constexpr counter_type cache_size() { return 4; }
inline constexpr counter_type cache_index(counter_type c) { return (3u & c); }

} // namespace cbprng
} // namespace arb

