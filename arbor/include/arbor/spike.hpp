#pragma once

#include <ostream>
#include <type_traits>
#include <functional>

#include <arbor/common_types.hpp>
#include <arbor/serdes.hpp>

namespace arb {

template <typename I>
struct basic_spike {
    using id_type = I;

    id_type source = id_type{};
    time_type time = -1;

    basic_spike() = default;

    basic_spike(id_type s, time_type t):
        source(std::move(s)), time(t)
    {}
    auto operator<=>(const basic_spike&) const = default;
    ARB_SERDES_ENABLE(basic_spike<I>, source, time);
};

/// Standard specialization:
using spike = basic_spike<cell_member_type>;

using spike_predicate = std::function<bool(const spike&)>;

// Custom stream operator for printing arb::spike<> values.
template <typename I>
std::ostream& operator<<(std::ostream& o, basic_spike<I> const& s) {
    return o << "S[src " << s.source << ", t " << s.time << "]";
}

} // namespace arb
