#pragma once

// Base class for parameter packs for GPU generated kernels:
// will be included by .cu generated sources.

#include <arbor/fvm_types.hpp>

namespace arb {
namespace gpu {

// Derived ppack structs may have ion_state_view fields:

struct ion_state_view {
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;

    value_type* current_density;
    value_type* reversal_potential;
    value_type* internal_concentration;
    value_type* external_concentration;
    value_type* ionic_charge;
};

// Parameter pack base:

struct mechanism_ppack_base {
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;
    using ion_state_view = ::arb::gpu::ion_state_view;

    index_type width_;

    const index_type* vec_ci_;
    const value_type* vec_t_;
    const value_type* vec_t_to_;
    const value_type* vec_dt_;
    const value_type* vec_v_;
    value_type* vec_i_;
    value_type* vec_g_;
    const value_type* temperature_degC_;
    const value_type* diam_um_;

    const index_type* node_index_;
    const index_type* multiplicity_;

    const value_type* weight_;
};

} // namespace gpu
} // namespace arb
