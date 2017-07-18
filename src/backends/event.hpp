#pragma once

#include <iostream>

#include <common_types.hpp>

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

} // namespace mc
} // namespace nest
