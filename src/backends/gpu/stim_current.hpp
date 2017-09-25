#pragma once

#include <backends/fvm_types.hpp>

namespace nest{
namespace mc{
namespace gpu {

void stim_current(
    const fvm_value_type* delay, const fvm_value_type* duration, const fvm_value_type* amplitude,
    const fvm_size_type* node_index, int n,
    const fvm_size_type* cell_index, const fvm_value_type* time, fvm_value_type* current);

} // namespace gpu
} // namespace mc
} // namespace nest
