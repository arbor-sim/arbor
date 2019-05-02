#pragma once

// Overloads for iostream formatted output for SIMD value classes.

#include <iostream>

#include <arbor/simd/simd.hpp>

namespace arb {
namespace simd {
namespace detail {

template <typename Impl>
std::ostream& operator<<(std::ostream& o, const simd_impl<Impl>& s) {
    using Simd = simd_impl<Impl>;

    typename Simd::scalar_type data[Simd::width];
    s.copy_to(data);
    o << data[0];
    for (unsigned i = 1; i<Simd::width; ++i) {
        o << ' ' << data[i];
    }
    return o;
}

} // namespace detail
} // namespace simd
} // namespace arb
