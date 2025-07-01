#pragma once

// Implementations of mathematical operations required by generated CUDA mechanisms and back-end methods.

#include <cfloat>
#include <cmath>

#include "gpu_api.hpp"

namespace arb {
namespace gpu {

__device__
constexpr inline double safeinv(double x) {
    // NOTE: To be checked for performance:
    //    return 1/((1.0 + x == 1.0) ? DBL_EPSILON : x);
    // generates branchless PTX and one less instruction.
    if (1.0 + x == 1.0) return 1/DBL_EPSILON;
    return 1/x;
}

__device__
constexpr inline double exprelr(double x) {
    if (1.0 + x == 1.0) return 1.0;
    return x/expm1(x);
}

// Return minimum of the two values
//
// NOTE: generic std::min doesn't generate CUDA intrinsics, but std::fmin does.
//       Thus, we special case those instances
template <typename T>
__device__
constexpr inline T min(T lhs, T rhs) {
    if constexpr (std::is_same_v<T, double>) return std::fmin(lhs, rhs);
    if constexpr (std::is_same_v<T, float>)  return std::fminf(lhs, rhs);
    return lhs < rhs ? lhs : rhs;
}

// Return maximum of the two values
//
// NOTE: generic std::max doesn't generate CUDA intrinsics, but std::fmax does.
//       Thus, we special case those instances
template <typename T>
__device__
constexpr inline T max(T lhs, T rhs) {
    if constexpr (std::is_same_v<T, double>) return std::fmax(lhs, rhs);
    if constexpr (std::is_same_v<T, float>)  return std::fmaxf(lhs, rhs);
    return lhs<rhs? rhs: lhs;
}

template <typename T>
__device__
constexpr inline T lerp(T a, T b, T u) {
    return std::fma(u, b, std::fma(-u, a, a));
}

constexpr double pi = 3.1415926535897932384626433832795l;

} // namespace gpu
} // namespace arb
