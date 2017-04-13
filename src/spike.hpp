#pragma once

#include <ostream>
#include <type_traits>

#include <common_types.hpp>

namespace nest {
namespace mc {

template <typename I, typename Time>
struct basic_spike {
    using id_type = I;
    using time_type = Time;

    id_type source = id_type{};
    time_type time = -1;

    basic_spike() = default;

    basic_spike(id_type s, time_type t):
        source(s), time(t)
    {}
};

/// Standard specialization:
using spike = basic_spike<cell_member_type, default_time_type>;

} // namespace mc
} // namespace nest

// Custom stream operator for printing nest::mc::spike<> values.
template <typename I, typename T>
std::ostream& operator<<(std::ostream& o, nest::mc::basic_spike<I, T> s) {
    return o << "spike[t " << s.time << ", src " << s.source << "]";
}

/// Less than comparison operator for nest::mc::spike<> values:
/// spikes are ordered by spike time, for use in sorting and queueing.
template <typename I, typename T>
bool operator<(nest::mc::basic_spike<I, T> lhs, nest::mc::basic_spike<I, T> rhs) {
    return lhs.time < rhs.time;
}


