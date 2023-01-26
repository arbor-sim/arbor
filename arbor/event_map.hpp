#pragma once

#include <vector>

#include "backends/event.hpp"
#include "util/container_map.hpp"

namespace arb {

using mechanism_event_map = util::container_map<
    cell_local_size_type,
    std::vector<deliverable_event>>;

using event_map = util::container_map<
    cell_local_size_type,
    mechanism_event_map>;

inline void add_event(event_map& m, time_type time, target_handle h, float weight) {
    m[h.mech_id][h.mech_index].emplace_back(time, h, weight);
}

inline void add_event(event_map& m, const deliverable_event& ev) {
    m[ev.handle.mech_id][ev.handle.mech_index].push_back(ev);
}

} // namespace arb
