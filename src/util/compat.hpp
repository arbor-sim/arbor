#pragma once

/* Collection of compatibility workarounds to deal with compiler defects */

#include <cstddef>
#include <cmath>

namespace compat {

// std::end() broken with (at least) xlC 13.1.4.

template <typename T>
auto end(T& x) -> decltype(x.end()) { return x.end(); }

template <typename T, std::size_t N>
T* end(T (&x)[N]) { return &x[0]+N; }

template <typename T, std::size_t N>
const T* end(const T (&x)[N]) { return &x[0]+N; }

// Work-around bad optimization reordering in xlC 13.1.4.

inline void compiler_barrier_if_xlc_leq(unsigned ver) {
#if defined(__xlC__)
    if (__xlC__<=ver) {
        asm volatile ("" ::: "memory");
    }
#endif
}

// Work-around a bad inlining-related optimization with icpc 16.0.3 and -xMIC-AVX512,

inline void compiler_barrier_if_icc_leq(unsigned ver) {
#if defined(__INTEL_COMPILER_BUILD_DATE)
    if (__INTEL_COMPILER_BUILD_DATE<=ver) {
        asm volatile ("" ::: "memory");
    }
#endif
}

// Work-around bad ordering of std::isinf() (sometimes) within switch, xlC 13.1.4;
// wrapping the call within another function appears to be sufficient.

template <typename X>
inline constexpr bool isinf(X x) { return std::isinf(x); }


} // namespace compat
