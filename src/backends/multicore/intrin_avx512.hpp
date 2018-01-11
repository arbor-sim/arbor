#pragma once

// vector types for avx512

// double precision avx512 register
using vecd_avx512  = __m512d;
// 8 way mask for avx512 register (for use with double precision)
using mask8_avx512 = __mmask8;

inline vecd_avx512 set(double x) {
    return _mm512_set1_pd(x);
}

namespace detail {
    // Useful constants in vector registers
    const vecd_avx512 vecd_avx512_zero = set(0.0);
    const vecd_avx512 vecd_avx512_one  = set(1.0);
    const vecd_avx512 vecd_avx512_two  = set(2.0);
    const vecd_avx512 vecd_avx512_nan  = set(std::numeric_limits<double>::quiet_NaN());
    const vecd_avx512 vecd_avx512_inf  = set(std::numeric_limits<double>::infinity());
    const vecd_avx512 vecd_avx512_ninf = set(-std::numeric_limits<double>::infinity());
}

//
// Operations on vector registers.
//
// shorter, less verbose wrappers around intrinsics
//

inline vecd_avx512 blend(mask8_avx512 m, vecd_avx512 x, vecd_avx512 y) {
    return _mm512_mask_blend_pd(m, x, y);
}

inline vecd_avx512 add(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_add_pd(x, y);
}

inline vecd_avx512 sub(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_sub_pd(x, y);
}

inline vecd_avx512 mul(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_mul_pd(x, y);
}

inline vecd_avx512 div(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_div_pd(x, y);
}

inline vecd_avx512 max(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_max_pd(x, y);
}

inline vecd_avx512 expm1(vecd_avx512 x) {
    // Assume that we are using the Intel compiler, and use the vectorized expm1
    // defined in the Intel SVML library.
    return _mm512_expm1_pd(x);
}

inline mask8_avx512 less(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_cmp_pd_mask(x, y, 0);
}

inline mask8_avx512 greater(vecd_avx512 x, vecd_avx512 y) {
    return _mm512_cmp_pd_mask(x, y, 30);
}

inline vecd_avx512 abs(vecd_avx512 x) {
    return max(x, sub(set(0.), x));
}

inline vecd_avx512 min(vecd_avx512 x, vecd_avx512 y) {
    // substitute values in x with values from y where x>y
    return blend(greater(x, y), x, y);
}

inline vecd_avx512 exprelr(vecd_avx512 x) {
    const auto ones = set(1);
    return blend(less(ones, add(x, ones)), div(x, expm1(x)), ones);
}
