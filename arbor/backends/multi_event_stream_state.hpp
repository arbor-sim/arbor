#pragma once

#include <arbor/fvm_types.hpp>

// Pointer representation of multi-event stream marked event state,
// common across CPU and GPU backends.

namespace arb {

template <typename EvData>
struct multi_event_stream_state {
    using value_type = EvData;

    fvm_size_type n;                      // number of streams
    const value_type* ev_data;            // array of event data items
    const fvm_index_type* begin_offset;   // array of offsets to beginning of marked events
    const fvm_index_type* end_offset;     // array of offsets to end of marked events

    fvm_size_type n_streams() const {
        return n;
    }

    const value_type* begin_marked(fvm_index_type i) const {
        return ev_data+begin_offset[i];
    }

    const value_type* end_marked(fvm_index_type i) const {
        return ev_data+end_offset[i];
    }
};

} // namespace arb
