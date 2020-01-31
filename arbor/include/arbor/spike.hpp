#pragma once

#include <ostream>
#include <type_traits>

#include <arbor/common_types.hpp>

namespace arb {

template <typename I>
struct basic_spike {
    using id_type = I;

    id_type source = id_type{};
    time_type time = -1;

    basic_spike() = default;

    basic_spike(id_type s, time_type t):
        source(s), time(t)
    {}

    friend bool operator==(const basic_spike& l, const basic_spike& r) {
        return l.time==r.time && l.source==r.source;
    }
};

/// Standard specialization:
using spike = basic_spike<cell_member_type>;

} // namespace arb

// Custom stream operator for printing arb::spike<> values.
template <typename I>
std::ostream& operator<<(std::ostream& o, arb::basic_spike<I> const& s) {
    return o << "S[src " << s.source << ", t " << s.time << "]";
}
