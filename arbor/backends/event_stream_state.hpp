#pragma once

#include <arbor/fvm_types.hpp>

// Pointer representation of multi-event stream marked event state,
// common across CPU and GPU backends.

namespace arb {

template <typename EvData>
struct event_stream_state {
    using value_type = EvData;

    const value_type* begin_marked;   // offset to beginning of marked events
    const value_type* end_marked;     // offset to end of marked events

    std::size_t size() const {
        return end_marked - begin_marked;
    }
};

} // namespace arb
