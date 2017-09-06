#pragma once

#include <common_types.hpp>
#include <backends/event.hpp>
#include <backends/fvm_types.hpp>

namespace nest {
namespace mc {
namespace gpu {

vid run_samples(fvm_size_type n, fvm_value_type* store, const raw_probe_info* data, const fvm_size_type* begin, const fvm_size_type* end);

} // namespace gpu
} // namespace mc
} // namespace nest

