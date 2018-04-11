#pragma once

// Base class for parameter packs for GPU generated kernels:
// will be included by .cu generated sources.

#include <backends/fvm_types.hpp>

namespace arb {
namespace gpu {

struct mechanism_ppack_base {
    const fvm_index_type* vec_ci_;
    const fvm_value_type* vec_t_;
    const fvm_value_type* vec_t_to_;
    const fvm_value_type* vec_dt_;
    const fvm_value_type* vec_v_;
    fvm_value_type* vec_i_;

    const fvm_index_type* node_index_;
    const fvm_value_type* weight_;
};

} // namespace gpu
} // namespace arb
