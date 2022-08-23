#pragma once

#include <arbor/export.hpp>
#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {

// Pointer representation of state is passed to GPU kernels.

struct istim_pp {
    // Stimulus constant and mutable data:
    const arb_index_type* accu_index;
    const arb_index_type* accu_to_cv;
    const arb_value_type* frequency;
    const arb_value_type* phase;
    const arb_value_type* envl_amplitudes;
    const arb_value_type* envl_times;
    const arb_index_type* envl_divs;
    arb_value_type* accu_stim;
    arb_index_type* envl_index;

    // Pointers to shared state data:
    const arb_value_type* time;
    const arb_index_type* cv_to_intdom;
    arb_value_type* current_density;
};

ARB_ARBOR_API void istim_add_current_impl(int n, const istim_pp& pp);

} // namespace gpu
} // namespace arb
