#pragma once

#include <type_traits>

#include "wrappers.hpp"

namespace nest {
namespace mc {
namespace memory {

template <typename LHS, typename RHS>
void copy(LHS&& from, RHS&& to) {
#ifndef NDEBUG
    assert(from.size() == to.size());
#endif
#ifdef VERBOSE
    std::cerr
        << util::blue("copy") << " "
        << util::pretty_printer<typename std::decay<LHS>::type>::print(from)
        << util::cyan(" -> ")
        << util::pretty_printer<typename std::decay<RHS>::type>::print(to)  << "\n";
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
    using lhs_type = typename std::decay<LHS>::type;
    static_assert(
        std::is_convertible<T, typename lhs_type::value_type>::value,
        "can't fill container with a value of the supplied type"
    );
#ifdef VERBOSE
    std::cerr
        << util::blue("fill") << " "
        << util::pretty_printer<typename std::decay<LHS>::type>::print(target)
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
} // namespace mc
} // namespace nest
