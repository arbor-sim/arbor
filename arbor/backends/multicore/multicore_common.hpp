#pragma once

// Storage classes and other common types across
// multicore back end implementations.
//
// Defines array, iarray, and specialized multi-event stream classes.

#include <utility>
#include <vector>

#include <arbor/fvm_types.hpp>

#include "backends/event.hpp"
#include "util/padded_alloc.hpp"

#include "multi_event_stream.hpp"

namespace arb {
namespace multicore {

template <typename V>
using padded_vector = std::vector<V, util::padded_allocator<V>>;

using array  = padded_vector<fvm_value_type>;
using iarray = padded_vector<fvm_index_type>;
using gjarray = padded_vector<fvm_gap_junction>;

using deliverable_event_stream = arb::multicore::multi_event_stream<deliverable_event>;
using sample_event_stream = arb::multicore::multi_event_stream<sample_event>;

} // namespace multicore
} // namespace arb

