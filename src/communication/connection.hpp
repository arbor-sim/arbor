#pragma once

#include <cstdint>

#include <event_queue.hpp>
#include <communication/spike.hpp>

namespace nest {
namespace mc {
namespace communication {

class connection {
public:
    using id_type = uint32_t;
    connection(id_type src, id_type dest, float w, float d) :
        source_(src),
        destination_(dest),
        weight_(w),
        delay_(d)
    {}

    float weight() const { return weight_; }
    float delay() const { return delay_; }

    id_type source() const { return source_; }
    id_type destination() const { return destination_; }

    postsynaptic_spike_event make_event(spike<id_type> s) {
        return {destination_, s.time + delay_, weight_};
    }

private:

    id_type source_;
    id_type destination_;
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
bool operator< (connection lhs, connection::id_type rhs) {
    return lhs.source() < rhs;
}

static inline
bool operator< (connection::id_type lhs, connection rhs) {
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
