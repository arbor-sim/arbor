#pragma once

#include <ostream>
#include <type_traits>

namespace nest {
namespace mc {

template <typename I, typename TimeT>
struct spike {
    using id_type = I;
    using time_type = TimeT;

    id_type source = id_type{};
    time_type time = -1.;

    spike() = default;

    spike(id_type s, time_type t) :
        source(s), time(t)
    {}
};

} // namespace mc
} // namespace nest

/// custom stream operator for printing nest::mc::spike<> values
template <typename I, typename T>
std::ostream& operator<<(std::ostream& o, nest::mc::spike<I, T> s) {
    return o << "spike[t " << s.time << ", src " << s.source << "]";
}

/// less than comparison operator for nest::mc::spike<> values
/// spikes are ordered by spike time, for use in sorting and queueing
template <typename I, typename T>
bool operator<(nest::mc::spike<I, T> lhs, nest::mc::spike<I, T> rhs) {
    return lhs.time < rhs.time;
}

