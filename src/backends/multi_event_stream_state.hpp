#pragma once

#include <common_types.hpp>
#include <util/cuda_decorate.hpp>

// Pointer representation of multi-event stream marked event state,
// common across CPU and GPU backends.

namespace nest {
namespace mc {

template <typename EvData>
struct multi_event_stream_state {
    using value_type = EvData;

    cell_size_type n;                     // number of streams
    const value_type* ev_data;            // array of event data items
    const cell_size_type* begin_offset;   // array of offsets to beginning of marked events
    const cell_size_type* end_offset;     // array of offsets to end of marked events

    CUDA_DECORATE
    fvm_size_type n_streams() const {
        return n;
    }

    CUDA_DECORATE
    const value_type* begin_marked(fvm_size_type i) const {
        return ev_data+begin_offset[i];
    }

    CUDA_DECORATE
    const value_type* end_marked(fvm_size_type i) const {
        return ev_data+end_offset[i];
    }
};

} // namespace nest
} // namespace mc
