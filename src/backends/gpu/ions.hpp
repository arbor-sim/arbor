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

// prototype for inializing ion species concentrations
void init_concentration(std::size_t n,
            fvm_value_type* Xi, fvm_value_type* Xo,
            const fvm_value_type* weight_Xi, const fvm_value_type* weight_Xo,
            fvm_value_type c_int, fvm_value_type c_ext);

} // namespace gpu
} // namespace arb

