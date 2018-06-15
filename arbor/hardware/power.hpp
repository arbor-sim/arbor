#pragma once

#include <cstdint>

namespace arb {
namespace hw {

// Test for support on configured architecture:
bool has_energy_measurement();

// Energy in Joules (J)
using energy_size_type = std::uint64_t;

// Returns energy_size_type(-1) if unable to read energy
energy_size_type energy();

} // namespace hw
} // namespace arb
