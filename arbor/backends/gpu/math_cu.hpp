#pragma once
#include <cfloat>

// Implementations of mathematical operations required
// by generated CUDA mechanisms.

namespace arb {
namespace gpu {

__device__
inline double safeinv(double x) {
    if (1.0+x == 1.0) {
        return 1/DBL_EPSILON;
    }
    return 1/x;
}

__device__
inline double exprelr(double x) {
    if (1.0+x == 1.0) {
        return 1.0;
    }
    return x/expm1(x);
}

// Return minimum of the two values
template <typename T>
__device__
inline T min(T lhs, T rhs) {
    return lhs<rhs? lhs: rhs;
}

// Return maximum of the two values
template <typename T>
__device__
inline T max(T lhs, T rhs) {
    return lhs<rhs? rhs: lhs;
}

} // namespace gpu
} // namespace arb
