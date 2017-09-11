#pragma once

#include <common_types.hpp>
#include "fvm_types.hpp"

// Structures for the representation of event delivery targets and
// staged events.

namespace nest {
namespace mc {

struct target_handle {
    cell_local_size_type mech_id;    // mechanism type identifier (per cell group).
    cell_local_size_type index;      // instance of the mechanism
    cell_size_type cell_index;       // which cell (acts as index into e.g. vec_t)

    target_handle() {}
    target_handle(cell_local_size_type mech_id, cell_local_size_type index, cell_size_type cell_index):
        mech_id(mech_id), index(index), cell_index(cell_index) {}
};

struct deliverable_event {
    time_type time;
    target_handle handle;
    float weight;

    deliverable_event() {}
    deliverable_event(time_type time, target_handle handle, float weight):
        time(time), handle(handle), weight(weight) {}
};

// Interface for access to the event stream data for mechanism gpu kernels.
struct gpu_event_state {
    using size_type = fvm_size_type;
    using value_type = fvm_value_type;
    size_type n;
    const size_type* ev_mech_id;
    const size_type* ev_index;
    const value_type* ev_weight;
    const size_type* span_begin;
    const size_type* mark;
};

} // namespace mc
} // namespace nest
