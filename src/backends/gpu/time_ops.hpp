#pragma once

#include <backends/fvm_types.hpp>

namespace nest {
namespace mc {
namespace gpu {

void update_time_to(fvm_size_type n,
                    fvm_value_type* time_to,
                    const fvm_value_type* time,
                    fvm_value_type dt,
                    fvm_value_type tmax);

void set_dt(fvm_size_type ncell,
            fvm_size_type ncomp,
            fvm_value_type* dt_cell,
            fvm_value_type* dt_comp,
            const fvm_value_type* time_to,
            const fvm_value_type* time,
            const fvm_size_type* cv_to_cell);

} // namespace gpu
} // namespace mc
} // namespace nest
