#pragma once

#include <arbor/fvm_types.hpp>

// Pointer representation of event stream marked event state,
// common across CPU and GPU backends.

namespace arb {

template <typename EvData>
struct event_stream_state {
    using value_type = EvData;

    const value_type* begin_marked;   // offset to beginning of marked events
    const value_type* end_marked;     // offset to one-past-end of marked events

    std::size_t size() const noexcept {
        return end_marked - begin_marked;
    }

    bool empty() const noexcept {
        return (size() == 0u);
    }
};

} // namespace arb
