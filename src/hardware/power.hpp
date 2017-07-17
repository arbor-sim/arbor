#pragma once

#include <cstdint>

namespace nest {
namespace mc {
namespace hw {

// Energy in Joules (J)
using energy_size_type = std::uint64_t;

// Returns negative value if unable to read energy
energy_size_type energy();

} // namespace hw
} // namespace mc
} // namespace nest
