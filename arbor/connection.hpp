#pragma once

#include <cstdint>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

namespace arb {

class connection {
public:
    connection() = default;
    connection(cell_member_type src,
               cell_lid_type dest,
               float w,
               float d,
               cell_gid_type didx=cell_gid_type(-1)):
        source(src),
        destination(dest),
        weight(w),
        delay(d),
        index_on_domain(didx)
    {}

    spike_event make_event(const spike& s) { return { destination, s.time + delay, weight}; }

    cell_member_type source;
    cell_lid_type destination;
    float weight;
    float delay;
    cell_size_type index_on_domain;
};

struct ext_connection {
    ext_guid_type source_;
    cell_lid_type destination_;
    float weight_;
    float delay_;
    cell_size_type index_on_domain_;
};

inline
spike_event make_event(const connection& c, const spike& s) {
    return {c.destination(), s.time + c.delay(), c.weight()};
}

inline
spike_event make_event(const ext_connection& c, const spike& s) {
    return {c.destination_, s.time + c.delay_, c.weight_};
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
