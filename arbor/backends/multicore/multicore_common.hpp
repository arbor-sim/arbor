#pragma once

// Storage classes and other common types across
// multicore back end implementations.
//
// Defines array, iarray, and specialized multi-event stream classes.

#include <vector>

#include <arbor/fvm_types.hpp>

#include "backends/event.hpp"
#include "util/padded_alloc.hpp"

#include "event_stream.hpp"

namespace arb {
namespace multicore {

template <typename V>
using padded_vector = std::vector<V, util::padded_allocator<V>>;

using array  = padded_vector<arb_value_type>;
using iarray = padded_vector<arb_index_type>;

} // namespace multicore
} // namespace arb

