#pragma once

#include <arbor/fvm_types.hpp>

namespace arb {

struct constraint_partition {
    size_t n_contiguous  = 0ul;
    size_t n_constant    = 0ul;
    size_t n_independent = 0ul;
    size_t n_none        = 0ul;
    fvm_index_type* contiguous  = nullptr;
    fvm_index_type* constant    = nullptr;
    fvm_index_type* independent = nullptr;
    fvm_index_type* none        = nullptr;
};

struct mechanism_ppack {
    fvm_index_type width_;
    fvm_index_type n_detectors_;
    const fvm_index_type* vec_ci_;
    const fvm_index_type* vec_di_;
    const fvm_value_type* vec_t_;
    const fvm_value_type* vec_dt_;
    const fvm_value_type* vec_v_;
    fvm_value_type* vec_i_;
    fvm_value_type* vec_g_;
    const fvm_value_type* temperature_degC_;
    const fvm_value_type* diam_um_;
    const fvm_value_type* time_since_spike_;
    const fvm_index_type* node_index_;
    const fvm_index_type* multiplicity_;
    const fvm_value_type* weight_;

    constraint_partition index_constraints_;
};
} // namespace arb
