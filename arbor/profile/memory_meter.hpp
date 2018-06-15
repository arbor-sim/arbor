#pragma once

#include <arbor/profile/meter.hpp>

namespace arb {
namespace profile {

meter_ptr make_memory_meter();
meter_ptr make_gpu_memory_meter();

} // namespace profile
} // namespace arb
