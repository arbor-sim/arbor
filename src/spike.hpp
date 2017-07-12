#pragma once

#include <ostream>
#include <type_traits>

#include <common_types.hpp>

namespace nest {
namespace mc {

template <typename I>
struct basic_spike {
    using id_type = I;

    id_type source = id_type{};
    time_type time = -1;

    basic_spike() = default;

    basic_spike(id_type s, time_type t):
        source(s), time(t)
    {}

    /// Less than comparison operator for nest::mc::spike<> values:
    /// spikes are ordered by spike time, for use in sorting and queueing.
    friend bool operator<(basic_spike lhs, basic_spike rhs) {
        return lhs.time < rhs.time;
    }

    friend bool operator<(cell_member_type lhs, basic_spike rhs) {
        return lhs < rhs.source;
    }

    friend bool operator<(basic_spike lhs, cell_member_type rhs) {
        return lhs.source < rhs;
    }
};

/// Standard specialization:
using spike = basic_spike<cell_member_type>;

} // namespace mc
} // namespace nest

// Custom stream operator for printing nest::mc::spike<> values.
template <typename I>
std::ostream& operator<<(std::ostream& o, nest::mc::basic_spike<I> s) {
    return o << "S[src " << s.source << ", t " << s.time << "]";
}
