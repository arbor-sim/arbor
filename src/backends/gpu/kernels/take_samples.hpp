#pragma once

#include <common_types.hpp>
#include <backends/event.hpp>
#include <backends/fvm_types.hpp>
#include <backends/multi_event_stream_state.hpp>

namespace nest {
namespace mc {
namespace gpu {

void take_samples(const multi_event_stream_state<raw_probe_info>& s, const fvm_value_type* time, fvm_value_type* sample_time, fvm_value_type* sample_value);

} // namespace gpu
} // namespace mc
} // namespace nest

