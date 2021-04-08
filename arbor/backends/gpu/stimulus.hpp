#pragma once

#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {

// Pointer representation of state is passed to GPU kernels.

struct istim_pp {
    // Stimulus constant and mutable data:
    const fvm_index_type* accu_index;
    const fvm_index_type* accu_to_cv;
    const fvm_value_type* frequency;
    const fvm_value_type* phase;
    const fvm_value_type* envl_amplitudes;
    const fvm_value_type* envl_times;
    const fvm_index_type* envl_divs;
    fvm_value_type* accu_stim;
    fvm_index_type* envl_index;

    // Pointers to shared state data:
    const fvm_value_type* time;
    const fvm_index_type* cv_to_intdom;
    fvm_value_type* current_density;
};

void istim_add_current_impl(int n, const istim_pp& pp);

} // namespace gpu
} // namespace arb
