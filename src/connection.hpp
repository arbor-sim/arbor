#pragma once

#include <cstdint>

#include <common_types.hpp>
#include <event_queue.hpp>
#include <spike.hpp>

namespace nest {
namespace mc {

template <typename Time>
class connection {
public:
    using id_type = cell_member_type;
    using time_type = Time;

    connection(id_type src, id_type dest, float w, time_type d) :
        source_(src),
        destination_(dest),
        weight_(w),
        delay_(d)
    {}

    float weight() const { return weight_; }
    float delay() const { return delay_; }

    id_type source() const { return source_; }
    id_type destination() const { return destination_; }

    postsynaptic_spike_event<time_type> make_event(spike<id_type, time_type> s) {
        return {destination_, s.time + delay_, weight_};
    }

private:
    id_type source_;
    id_type destination_;
    float weight_;
    time_type delay_;
};

// connections are sorted by source id
// these operators make for easy interopability with STL algorithms

template <typename T>
static inline bool operator<(connection<T> lhs, connection<T> rhs) {
    return lhs.source() < rhs.source();
}

template <typename T>
static inline bool operator<(connection<T> lhs, typename connection<T>::id_type rhs) {
    return lhs.source() < rhs;
}

template <typename T>
static inline bool operator<(typename connection<T>::id_type lhs, connection<T> rhs) {
    return lhs < rhs.source();
}

} // namespace mc
} // namespace nest

template <typename T>
static inline std::ostream& operator<<(std::ostream& o, nest::mc::connection<T> const& con) {
    return o << "con [" << con.source() << " -> " << con.destination()
             << " : weight " << con.weight()
             << ", delay " << con.delay() << "]";
}
