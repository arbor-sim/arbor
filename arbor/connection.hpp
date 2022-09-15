#pragma once

#include <cstdint>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

namespace arb {

struct connection {
    cell_member_type source = {0, 0};
    cell_lid_type destination = 0;
    float weight = 0.0f;
    float delay = 0.0f;
    cell_size_type index_on_domain = cell_gid_type(-1);
};

inline
bool is_external(cell_gid_type gid) {
    auto msb = sizeof(cell_gid_type)*8 - 1;
    return bool(gid & (1 << msb));
}

inline
bool is_external(const connection& c) {
    return is_external(c.source.gid);
}

inline
spike_event make_event(const connection& c, const spike& s) {
    return {c.destination, s.time + c.delay, c.weight};
}

// connections are sorted by source id
// these operators make for easy interopability with STL algorithms
static inline bool operator<(const connection& lhs, const connection& rhs) { return lhs.source < rhs.source; }
static inline bool operator<(const connection& lhs, cell_member_type rhs)  { return lhs.source < rhs; }
static inline bool operator<(cell_member_type lhs, const connection& rhs)  { return lhs < rhs.source; }

} // namespace arb

static inline std::ostream& operator<<(std::ostream& o, arb::connection const& con) {
    return o << "con [" << con.source << " -> " << con.destination
             << " : weight " << con.weight
             << ", delay " << con.delay
             << ", index " << con.index_on_domain << "]";
}
