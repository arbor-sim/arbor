#pragma once

/* Collection of compatibility workarounds to deal with compiler defects */

// Note: workarounds for xlC removed; use of xlC to build Arbor is deprecated.

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

// Work-around a bad inlining-related optimization with icpc 16.0.3 and -xMIC-AVX512,

inline void compiler_barrier_if_icc_leq(unsigned ver) {
#if defined(__INTEL_COMPILER_BUILD_DATE)
    if (__INTEL_COMPILER_BUILD_DATE<=ver) {
        asm volatile ("" ::: "memory");
    }
#endif
}

// Work-around for bad vectorization of fma in gcc version < 8.2

template <typename T>
#if defined(__GNUC__) && (100*__GNUC__ + __GNUC_MINOR__ < 802)
__attribute((optimize("no-tree-vectorize")))
#endif
inline auto fma(T a, T b, T c) {
    return std::fma(a, b, c);
}

} // namespace compat
