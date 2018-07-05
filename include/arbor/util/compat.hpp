#pragma once

/* Collection of compatibility workarounds to deal with compiler defects */

#include <cstddef>
#include <cmath>

namespace compat {

constexpr bool using_intel_compiler(int major=0, int minor=0, int patchlevel=0) {
#if defined(__INTEL_COMPILER)
    return __INTEL_COMPILER >= major*100 + minor &&
           __INTEL_COMPILER_UPDATE >= patchlevel;
#else
    return false;
#endif
}

constexpr bool using_gnu_compiler(int major=0, int minor=0, int patchlevel=0) {
#if defined(__GNUC__)
    return (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
            > (major*10000 + minor*100 + patchlevel);
#else
    return false;
#endif
}

// std::end() broken with xlC 13.1.4; fixed in 13.1.5.

namespace impl {
    using std::end;
    template <typename T>
    auto end_(T& x) -> decltype(end(x)) { return end(x); }
}

template <typename T>
auto end(T& x) -> decltype(impl::end_(x)) { return impl::end_(x); }

template <typename T, std::size_t N>
T* end(T (&x)[N]) { return &x[0]+N; }

template <typename T, std::size_t N>
const T* end(const T (&x)[N]) { return &x[0]+N; }

// Work-around bad optimization reordering in xlC 13.1.4.
// Note: still broken in xlC 14.1.0

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
// Note: still broken in xlC 14.1.0.

template <typename X>
inline constexpr bool isinf(X x) { return std::isinf(x); }

} // namespace compat
