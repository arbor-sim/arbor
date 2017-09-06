#pragma once

#include <backends/fvm_types.hpp>

#include "stack.hpp"

namespace nest {
namespace mc {
namespace gpu {

extern void test_thresholds(
    const fvm_size_type* cv_to_cell, const fvm_value_type* t_after, const fvm_value_type* t_before,
    int size,
    stack_base<threshold_crossing>& stack,
    fvm_size_type* is_crossed, fvm_value_type* prev_values,
    const fvm_size_type* cv_index, const fvm_value_type* values, const fvm_value_type* thresholds);

} // namespace gpu
} // namespace mc
} // namespace nest
