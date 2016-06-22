#pragma once

#include <type_traits>
#include <ostream>

namespace nest {
namespace mc {
namespace communication {

template <
    typename I,
    typename = typename std::enable_if<std::is_integral<I>::value>
>
struct spike {
    using index_type = I;
    index_type source = 0;
    float time = -1.;

    spike() = default;

    spike(index_type s, float t)
    :   source(s), time(t)
    {}
};

} // namespace mc
} // namespace nest
} // namespace communication

/// custom stream operator for printing nest::mc::spike<> values
template <typename I>
std::ostream& operator <<(std::ostream& o, nest::mc::communication::spike<I> s) {
    return o << "spike[t " << s.time << ", src " << s.source << "]";
}

/// less than comparison operator for nest::mc::spike<> values
/// spikes are ordered by spike time, for use in sorting and queueing
template <typename I>
bool operator <(
    nest::mc::communication::spike<I> lhs,
    nest::mc::communication::spike<I> rhs)
{
    return lhs.time < rhs.time;
}

