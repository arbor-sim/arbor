#pragma once

#include <cstdint>

namespace nest {
namespace mc {
namespace util {

#ifdef NMC_HAVE_CRAY
    constexpr bool has_power_measurement = true;
#else
    constexpr bool has_power_measurement = false;
#endif

// Energy in Joules (J)
using energy_size_type = std::uint64_t;

// Returns negative value if unable to read energy
energy_size_type energy();

} // namespace util
} // namespace mc
} // namespace nest
