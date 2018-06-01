#pragma once

#include <backends/gpu/mechanism_ppack_base.hpp>

namespace arb {
namespace gpu {

struct stimulus_pp: mechanism_ppack_base {
    fvm_value_type* delay;
    fvm_value_type* duration;
    fvm_value_type* amplitude;
};

void stimulus_current_impl(int n, const stimulus_pp&);

} // namespace gpu
} // namespace arb
