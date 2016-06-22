#pragma once

#include <cstdint>

#include <event_queue.hpp>
#include <communication/spike.hpp>

namespace nest {
namespace mc {
namespace communication {

class connection {
public:
    using index_type = uint32_t;
    connection(index_type src, index_type dest, float w, float d)
    :   source_(src),
        destination_(dest),
        weight_(w),
        delay_(d)
    { }

    float weight() const {
        return weight_;
    }
    float delay() const {
        return delay_;
    }

    index_type source() const {
        return source_;
    }
    index_type destination() const {
        return destination_;
    }

    local_event make_event(spike<index_type> s) {
        return {destination_, s.time + delay_, weight_};
    }

private:

    index_type source_;
    index_type destination_;
    float weight_;
    float delay_;
};

// connections are sorted by source id
// these operators make for easy interopability with STL algorithms

static inline
bool operator< (connection lhs, connection rhs) {
    return lhs.source() < rhs.source();
}

static inline
bool operator< (connection lhs, connection::index_type rhs) {
    return lhs.source() < rhs;
}

static inline
bool operator< (connection::index_type lhs, connection rhs) {
    return lhs < rhs.source();
}

} // namespace communication
} // namespace mc
} // namespace nest

static inline
std::ostream& operator<<(std::ostream& o, nest::mc::communication::connection const& con) {
    char buff[512];
    snprintf(
        buff, sizeof(buff), "con [%10u -> %10u : weight %8.4f, delay %8.4f]",
        con.source(), con.destination(), con.weight(), con.delay()
    );
    return o << buff;
}
