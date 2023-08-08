#pragma once

#include <arbor/common_types.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/serdes.hpp>
#include <arbor/mechanism_abi.h>
#include <arbor/generic_event.hpp>

// Structures for the representation of event delivery targets and
// staged events.

namespace arb {

// Post-synaptic spike events

struct target_handle {
    cell_local_size_type mech_id;    // mechanism type identifier (per cell group).
    cell_local_size_type mech_index; // instance of the mechanism

    target_handle() = default;

    target_handle(cell_local_size_type mech_id, cell_local_size_type mech_index):
        mech_id(mech_id), mech_index(mech_index) {}

    ARB_SERDES_ENABLE(target_handle, mech_id, mech_index);
};

}

template<typename K>
void serialize(arb::serializer &ser, const K &k, const arb::target_handle&);
template<typename K>
void deserialize(arb::serializer &ser, const K &k, arb::target_handle&);

namespace arb {

struct deliverable_event {
    time_type time = 0;
    float weight = 0;
    target_handle handle;

    deliverable_event() = default;
    deliverable_event(time_type time, target_handle handle, float weight):
        time(time), weight(weight), handle(handle) {}

    ARB_SERDES_ENABLE(deliverable_event, time, weight, handle);
};

template<>
struct has_event_index<deliverable_event> : public std::true_type {};

// Subset of event information required for mechanism delivery.
struct deliverable_event_data {
    cell_local_size_type mech_id;    // same as target_handle::mech_id
    cell_local_size_type mech_index; // same as target_handle::mech_index
    float weight;
    ARB_SERDES_ENABLE(deliverable_event_data, mech_id, mech_index, weight);
};

// Stream index accessor function for multi_event_stream:
inline cell_local_size_type event_index(const arb_deliverable_event_data& ed) {
    return ed.mech_index;
}

// Delivery data accessor function for multi_event_stream:
inline arb_deliverable_event_data event_data(const deliverable_event& ev) {
    return {ev.handle.mech_index, ev.weight};
}

inline arb_deliverable_event_stream make_event_stream_state(arb_deliverable_event_data* begin,
                                                            arb_deliverable_event_data* end) {
    return {begin, end};
}

// Sample events (raw values from back-end state).

using probe_handle = const arb_value_type*;

struct raw_probe_info {
    probe_handle handle;      // where the to-be-probed value sits
    sample_size_type offset;  // offset into array to store raw probed value
};

struct sample_event {
    time_type time;
    raw_probe_info raw;           // event payload: what gets put where on sample
};

inline raw_probe_info event_data(const sample_event& ev) {
    return ev.raw;
}

} // namespace arb
