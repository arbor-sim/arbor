#pragma once

/* 
 * Presents a half-open interval [a,b) of integral values as a container.
 */

#include <type_traits>
#include <utility>

#include <util/counter.hpp>
#include <util/range.hpp>

namespace arb {
namespace util {

template <typename I>
using span = range<counter<I>>;

template <typename I, typename J>
span<typename std::common_type<I, J>::type> make_span(I left, J right) {
    return span<typename std::common_type<I, J>::type>(left, right);
}

template <typename I, typename J>
span<typename std::common_type<I, J>::type> make_span(std::pair<I, J> interval) {
    return span<typename std::common_type<I, J>::type>(interval.first, interval.second);
}


} // namespace util
} // namespace arb
