#pragma once

// Storage classes and other common types across
// gpu back-end implementations.
//
// Defines array, iarray, and specialized multi-event stream classes.

#include <arbor/fvm_types.hpp>

#include "memory/memory.hpp"
#include "backends/event.hpp"
#include "backends/gpu/event_stream.hpp"

namespace arb {
namespace gpu {

using array  = memory::device_vector<arb_value_type>;
using iarray = memory::device_vector<arb_index_type>;
using sarray = memory::device_vector<arb_size_type>;

} // namespace gpu
} // namespace arb

