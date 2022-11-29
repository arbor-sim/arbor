#pragma once

#include <arbor/fvm_types.hpp>

// Pointer representation of event stream marked event state,
// common across CPU and GPU backends.

namespace arb {

template <typename EvData>
struct event_stream_state {
    using value_type = EvData;

    const value_type* data;
    const arb_size_type* begin_marked;
    const arb_size_type* end_marked;
    const arb_size_type kinds;
    const arb_size_type marked;

    std::size_t size() const { return marked; }
};

} // namespace arb
