#pragma once

#include <cstdint>

namespace nest {
namespace mc {
namespace util {

// Energy in Joules (J)
using energy_size_type = std::uint64_t;

// Returns negative value if unable to read energy
energy_size_type energy();

} // namespace util
} // namespace mc
} // namespace nest
