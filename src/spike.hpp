#pragma once

#include <ostream>
#include <type_traits>

namespace nest {
namespace mc {

template <typename I>
struct spike {
    using id_type = I;
    id_type source = id_type{};
    float time = -1.;

    spike() = default;

    spike(id_type s, float t) :
        source(s), time(t)
    {}
};

} // namespace mc
} // namespace nest

/// custom stream operator for printing nest::mc::spike<> values
template <typename I>
std::ostream& operator<<(std::ostream& o, nest::mc::spike<I> s) {
    return o << "spike[t " << s.time << ", src " << s.source << "]";
}

/// less than comparison operator for nest::mc::spike<> values
/// spikes are ordered by spike time, for use in sorting and queueing
template <typename I>
bool operator<(nest::mc::spike<I> lhs, nest::mc::spike<I> rhs) {
    return lhs.time < rhs.time;
}

