#pragma once

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

namespace arb {

struct connection {
    cell_member_type source = {0, 0};
    cell_lid_type target = 0;
    float weight = 0.0f;
    float delay = 0.0f;
    cell_size_type index_on_domain = cell_gid_type(-1);

    bool operator==(const connection&) const = default;

    // connections are sorted by source id
    // these operators make for easy interopability with STL algorithms
    auto operator<=>(const connection& rhs) const { return source <=> rhs.source; }
    auto operator<=>(const cell_member_type& rhs) const  { return source <=> rhs; }
};

inline
spike_event make_event(const connection& c, const spike& s) {
    return {c.target, s.time + c.delay, c.weight};
}

} // namespace arb

static inline std::ostream& operator<<(std::ostream& o, arb::connection const& con) {
    return o << "con [" << con.source << " -> " << con.target
             << " : weight " << con.weight
             << ", delay " << con.delay
             << ", index " << con.index_on_domain << "]";
}
