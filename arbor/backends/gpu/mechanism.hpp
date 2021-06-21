#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechanism_abi.h>

#include "backends/gpu/fvm.hpp"
#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

// Base class for all generated mechanisms for gpu back-end.
class mechanism: public arb::mechanism {
protected:
    fvm_size_type width_padded_ = 0;            // Width rounded up to multiple of pad/alignment.

    memory::device_vector<arb_value_type*> parameters_d_;
    memory::device_vector<arb_value_type*> state_vars_d_;
    memory::device_vector<arb_ion_state>   ion_states_d_;
};
} // namespace gpu
} // namespace arb
