#pragma once

#include <cstdint>

#include <backends/fvm_types.hpp>

namespace arb {
namespace gpu {

// prototype for nernst equation cacluation
void nernst(std::size_t n, int valency,
            fvm_value_type temperature,
            const fvm_value_type* Xo,
            const fvm_value_type* Xi,
            fvm_value_type* eX);

} // namespace gpu
} // namespace arb

