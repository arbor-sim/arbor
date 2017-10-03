#pragma once

#include <cstdint>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <spike.hpp>

namespace arb {

class connection {
public:
    connection() = default;
    connection( cell_member_type src,
                cell_member_type dest,
                float w,
                time_type d,
                cell_gid_type didx=cell_gid_type(-1)):
        source_(src),
        destination_(dest),
        weight_(w),
        delay_(d),
        domain_index_(didx)
    {}

    float weight() const { return weight_; }
    time_type delay() const { return delay_; }

    cell_member_type source() const { return source_; }
    cell_member_type destination() const { return destination_; }
    cell_gid_type domain_index() const { return domain_index_; }

    postsynaptic_spike_event make_event(const spike& s) {
        return {destination_, s.time + delay_, weight_};
    }

private:
    cell_member_type source_;
    cell_member_type destination_;
    float weight_;
    time_type delay_;
    cell_gid_type domain_index_;
};

// connections are sorted by source id
// these operators make for easy interopability with STL algorithms

static inline bool operator<(const connection& lhs, const connection& rhs) {
    return lhs.source() < rhs.source();
}

static inline bool operator<(const connection& lhs, cell_member_type rhs) {
    return lhs.source() < rhs;
}

static inline bool operator<(cell_member_type lhs, const connection& rhs) {
    return lhs < rhs.source();
}

} // namespace arb

static inline std::ostream& operator<<(std::ostream& o, arb::connection const& con) {
    return o << "con [" << con.source() << " -> " << con.destination()
             << " : weight " << con.weight()
             << ", delay " << con.delay() << "]";
}
