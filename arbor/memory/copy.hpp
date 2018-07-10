#pragma once

#include <type_traits>

#include <arbor/assert.hpp>

#include "wrappers.hpp"

namespace arb {
namespace memory {

template <typename LHS, typename RHS>
void copy(LHS&& from, RHS&& to) {
    arb_assert(from.size() == to.size());
#ifdef VERBOSE
    std::cerr
        << util::blue("copy") << " "
        << util::pretty_printer<std::decay_t<LHS>>::print(from)
        << util::cyan(" -> ")
        << util::pretty_printer<std::decay_t<RHS>>::print(to)  << "\n";
#endif
    // adapt views to the inputs
    auto lhs = make_const_view(from);
    auto rhs = make_view(to);

    // get a copy of the source view's coordinator
    typename decltype(lhs)::coordinator_type coord;
    // perform the copy
    coord.copy(lhs, rhs);
}

template <typename LHS, typename T>
void fill(LHS&& target, T value) {
    using lhs_type = std::decay_t<LHS>;
    static_assert(
        std::is_convertible<T, typename lhs_type::value_type>::value,
        "can't fill container with a value of the supplied type"
    );
#ifdef VERBOSE
    std::cerr
        << util::blue("fill") << " "
        << util::pretty_printer<std::decay_t<LHS>>::print(target)
        << util::cyan(" <- ")
        << T(value) << "\n";
#endif
    // adapt view to the inputs
    auto lhs = make_view(target);

    // get a copy of the target view's coordinator
    typename decltype(lhs)::coordinator_type coord;

    // perform the fill
    coord.set(lhs, T{value});
}

} // namespace memory
} // namespace arb
