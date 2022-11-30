#pragma once

#include <backends/multi_event_stream.hpp>

namespace arb {
namespace multicore {

template <typename Event>
using multi_event_stream = ::arb::multi_event_stream<Event>;

} // namespace multicore
} // namespace arb
