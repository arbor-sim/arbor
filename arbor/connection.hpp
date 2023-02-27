#pragma once

#include <cstdint>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

namespace arb {

class connection {
public:
    connection() = default;
    connection(cell_member_type src, cell_member_type dest, float w, float d):
        source(src),
        destination(dest),
        weight(w),
        delay(d) {}

    spike_event make_event(const spike& s) { return {destination.index, s.time + delay, weight}; }

    cell_member_type source;
    cell_member_type destination;
    float weight;
    float delay;
};

// connections are sorted by source id
// these operators make for easy interopability with STL algorithms
static inline bool operator<(const connection& lhs, const connection& rhs) { return lhs.source < rhs.source; }
static inline bool operator<(const connection& lhs, cell_member_type rhs)  { return lhs.source < rhs; }
static inline bool operator<(cell_member_type lhs, const connection& rhs)  { return lhs < rhs.source; }

} // namespace arb

static inline std::ostream& operator<<(std::ostream& o, arb::connection const& con) {
    return o << "con [" << con.source << " -> " << con.destination << " : weight " << con.weight
             << ", delay " << con.delay << "]";
}
