#pragma once

#include <backends/event_stream.hpp>

namespace arb {
namespace multicore {

template <typename Event>
using event_stream = ::arb::event_stream<Event>;

} // namespace multicore
} // namespace arb
