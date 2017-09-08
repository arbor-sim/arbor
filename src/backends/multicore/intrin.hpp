//
// Custom transcendental intrinsics
//
// Implementation inspired by the Cephes library:
//    - http://www.netlib.org/cephes/

#pragma once

#include <iostream>
#include <immintrin.h>

namespace nest {
namespace mc {
namespace multicore {

namespace detail {

constexpr double exp_limit = 708;

// P/Q polynomial coefficients for the exponential function
constexpr double P0exp = 9.99999999999999999910E-1;
constexpr double P1exp = 3.02994407707441961300E-2;
constexpr double P2exp = 1.26177193074810590878E-4;

constexpr double Q0exp = 2.00000000000000000009E0;
constexpr double Q1exp = 2.27265548208155028766E-1;
constexpr double Q2exp = 2.52448340349684104192E-3;
constexpr double Q3exp = 3.00198505138664455042E-6;

// P/Q polynomial coefficients for the log function
constexpr double P0log = 7.70838733755885391666E0;
constexpr double P1log = 1.79368678507819816313e1;
constexpr double P2log = 1.44989225341610930846e1;
constexpr double P3log = 4.70579119878881725854e0;
constexpr double P4log = 4.97494994976747001425e-1;
constexpr double P5log = 1.01875663804580931796e-4;

constexpr double Q0log = 2.31251620126765340583E1;
constexpr double Q1log = 7.11544750618563894466e1;
constexpr double Q2log = 8.29875266912776603211e1;
constexpr double Q3log = 4.52279145837532221105e1;
constexpr double Q4log = 1.12873587189167450590e1;
constexpr double ln2inv = 1.4426950408889634073599; // 1/ln(2)

// C1 + C2 = ln(2)
constexpr double C1 = 6.93145751953125E-1;
constexpr double C2 = 1.42860682030941723212E-6;

// C4 - C3 = ln(2)
constexpr double C3 = 2.121944400546905827679e-4;
constexpr double C4 = 0.693359375;

constexpr uint64_t dmant_mask = ((1UL<<52) - 1) | (1UL << 63); // mantissa + sign
constexpr uint64_t dexp_mask  = ((1UL<<11) - 1) << 52;
constexpr int exp_bias = 1023;
constexpr double dsqrth = 0.70710678118654752440;

// Useful constants in vector registers
const __m256d nmc_m256d_zero = _mm256_set1_pd(0.0);
const __m256d nmc_m256d_one  = _mm256_set1_pd(1.0);
const __m256d nmc_m256d_two  = _mm256_set1_pd(2.0);
const __m256d nmc_m256d_nan  = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
const __m256d nmc_m256d_inf  = _mm256_set1_pd(std::numeric_limits<double>::infinity());
const __m256d nmc_m256d_ninf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
}

static void nmc_mm256_print_pd(__m256d x, const char *name) __attribute__ ((unused));
static void nmc_mm256_print_epi32(__m128i x, const char *name) __attribute__ ((unused));
static void nmc_mm256_print_epi64x(__m256i x, const char *name) __attribute__ ((unused));
static __m256d nmc_mm256_exp_pd(__m256d x) __attribute__ ((unused));
static __m256d nmc_mm256_subnormal_pd(__m256d x) __attribute__ ((unused));
static __m256d nmc_mm256_frexp_pd(__m256d x, __m128i *e) __attribute__ ((unused));
static __m256d nmc_mm256_log_pd(__m256d x) __attribute__ ((unused));
static __m256d nmc_mm256_pow_pd(__m256d x, __m256d y) __attribute__ ((unused));

void nmc_mm256_print_pd(__m256d x, const char *name) {
    double *val = (double *) &x;
    std::cout << name << " = { ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << val[i] << " ";
    }

    std::cout << "}\n";
}

void nmc_mm256_print_epi32(__m128i x, const char *name) {
    int *val = (int *) &x;
    std::cout << name << " = { ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << val[i] << " ";
    }

    std::cout << "}\n";
}

void nmc_mm256_print_epi64x(__m256i x, const char *name) {
    uint64_t *val = (uint64_t *) &x;
    std::cout << name << " = { ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << val[i] << " ";
    }

    std::cout << "}\n";
}

//
// Calculates exponential using AVX2 instructions
//
//  Exponential is calculated as follows:
//     e^x = e^g * 2^n,
//
//  where g in [-0.5, 0.5) and n is an integer. Obviously 2^n can be calculated
//  fast with bit shift, whereas e^g is approximated using the following Pade'
//  form:
//
//     e^g = 1 + 2*g*P(g^2) / (Q(g^2)-P(g^2))
//
//  The exponents n and g are calculated using the following formulas:
//
//  n = floor(x/ln(2) + 0.5)
//  g = x - n*ln(2)
//
//  They can be derived as follows:
//
//    e^x = 2^(x/ln(2))
//        = 2^-0.5 * 2^(x/ln(2) + 0.5)
//        = 2^r'-0.5 * 2^floor(x/ln(2) + 0.5)     (1)
//
// Setting n = floor(x/ln(2) + 0.5),
//
//    r' = x/ln(2) - n, and r' in [0, 1)
//
// Substituting r' in (1) gives us
//
//    e^x = 2^(x/ln(2) - n) * 2^n, where x/ln(2) - n is now in [-0.5, 0.5)
//        = e^(x-n*ln(2)) * 2^n
//        = e^g * 2^n, where g = x - n*ln(2)      (2)
//
// NOTE: The calculation of ln(2) in (2) is split in two operations to
// compensate for rounding errors:
//
//   ln(2) = C1 + C2, where
//
//   C1 = floor(2^k*ln(2))/2^k
//   C2 = ln(2) - C1
//
// We use k=32, since this is what the Cephes library does historically.
// Theoretically, we could use k=52 to match the IEEE-754 double accuracy, but
// the standard library seems not to do that, so we are getting differences
// compared to std::exp() for large exponents.
//
__m256d nmc_mm256_exp_pd(__m256d x) {
    __m256d x_orig = x;

    __m256d px = _mm256_floor_pd(
        _mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(detail::ln2inv), x),
            _mm256_set1_pd(0.5)
        )
    );

    __m128i n = _mm256_cvtpd_epi32(px);

    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(detail::C1)));
    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(detail::C2)));

    __m256d xx = _mm256_mul_pd(x, x);

    // Compute the P and Q polynomials.

    // Polynomials are computed in factorized form in order to reduce the total
    // numbers of operations:
    //
    // P(x) = P0 + P1*x + P2*x^2 = P0 + x*(P1 + x*P2)
    // Q(x) = Q0 + Q1*x + Q2*x^2 + Q3*x^3 = Q0 + x*(Q1 + x*(Q2 + x*Q3))

    // Compute x*P(x**2)
    px = _mm256_set1_pd(detail::P2exp);
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P1exp));
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P0exp));
    px = _mm256_mul_pd(px, x);


    // Compute Q(x**2)
    __m256d qx = _mm256_set1_pd(detail::Q3exp);
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q2exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q1exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q0exp));

    // Compute 1 + 2*P(x**2) / (Q(x**2)-P(x**2))
    x = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
    x = _mm256_add_pd(detail::nmc_m256d_one,
                      _mm256_mul_pd(detail::nmc_m256d_two, x));

    // Finally, compute x *= 2**n
    __m256i n64 = _mm256_cvtepi32_epi64(n);
    n64 = _mm256_add_epi64(n64, _mm256_set1_epi64x(1023));
    n64 = _mm256_sll_epi64(n64, _mm_set_epi64x(0, 52));
    x = _mm256_mul_pd(x, _mm256_castsi256_pd(n64));

    // Treat exceptional cases
    __m256d is_large = _mm256_cmp_pd(
        x_orig, _mm256_set1_pd(detail::exp_limit), 30 /* _CMP_GT_OQ */
    );
    __m256d is_small = _mm256_cmp_pd(
        x_orig, _mm256_set1_pd(-detail::exp_limit), 17 /* _CMP_LT_OQ */
    );
    __m256d is_nan = _mm256_cmp_pd(x_orig, x_orig, 3 /* _CMP_UNORD_Q */ );

    x = _mm256_blendv_pd(x, detail::nmc_m256d_inf, is_large);
    x = _mm256_blendv_pd(x, detail::nmc_m256d_zero, is_small);
    x = _mm256_blendv_pd(x, detail::nmc_m256d_nan, is_nan);
    return x;

}

__m256d nmc_mm256_subnormal_pd(__m256d x) {
    __m256i x_raw = _mm256_castpd_si256(x);
    __m256i exp_mask = _mm256_set1_epi64x(detail::dexp_mask);
    __m256d x_exp = _mm256_castsi256_pd(_mm256_and_si256(x_raw, exp_mask));

    // Subnormals have a zero exponent
    return _mm256_cmp_pd(x_exp, detail::nmc_m256d_zero, 0 /* _CMP_EQ_OQ */);
}

__m256d nmc_mm256_frexp_pd(__m256d x, __m128i *e) {
    __m256i exp_mask  = _mm256_set1_epi64x(detail::dexp_mask);
    __m256i mant_mask = _mm256_set1_epi64x(detail::dmant_mask);

    __m256d x_orig = x;

    // we will work on the raw bits of x
    __m256i x_raw  = _mm256_castpd_si256(x);
    __m256i x_exp  = _mm256_and_si256(x_raw, exp_mask);
    x_exp = _mm256_srli_epi64(x_exp, 52);

    // We need bias-1 since frexp returns base values in (-1, -0.5], [0.5, 1)
    x_exp = _mm256_sub_epi64(x_exp, _mm256_set1_epi64x(detail::exp_bias-1));

    // IEEE-754 floats are in 1.<mantissa> form, but frexp needs to return a
    // float in (-1, -0.5], [0.5, 1). We convert x_ret in place by adding it
    // an 2^-1 exponent, i.e., 1022 in IEEE-754 format
    __m256i x_ret = _mm256_and_si256(x_raw, mant_mask);

    __m256i exp_bits = _mm256_slli_epi64(_mm256_set1_epi64x(detail::exp_bias-1), 52);
    x_ret = _mm256_or_si256(x_ret, exp_bits);
    x = _mm256_castsi256_pd(x_ret);

    // Treat special cases
    __m256d is_zero = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_zero, 0 /* _CMP_EQ_OQ */
    );
    __m256d is_inf = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_inf, 0 /* _CMP_EQ_OQ */
    );
    __m256d is_ninf = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_ninf, 0 /* _CMP_EQ_OQ */
    );
    __m256d is_nan = _mm256_cmp_pd(x_orig, x_orig, 3 /* _CMP_UNORD_Q */ );

    // Denormalized numbers have a zero exponent. Here we expect -1022 since we
    // have already prepared it as a power of 2
    __m256i is_denorm = _mm256_cmpeq_epi64(x_exp, _mm256_set1_epi64x(-1022));

    x = _mm256_blendv_pd(x, detail::nmc_m256d_zero, is_zero);
    x = _mm256_blendv_pd(x, detail::nmc_m256d_inf, is_inf);
    x = _mm256_blendv_pd(x, detail::nmc_m256d_ninf, is_ninf);
    x = _mm256_blendv_pd(x, detail::nmc_m256d_nan, is_nan);

    // FIXME: We treat denormalized numbers as zero here
    x = _mm256_blendv_pd(x, detail::nmc_m256d_zero,
                         _mm256_castsi256_pd(is_denorm));
    x_exp = _mm256_blendv_epi8(x_exp, _mm256_set1_epi64x(0), is_denorm);

    x_exp = _mm256_blendv_epi8(x_exp, _mm256_set1_epi64x(0),
                               _mm256_castpd_si256(is_zero));


    // We need to "compress" x_exp into the first 128 bits before casting it
    // safely to __m128i and return to *e
    x_exp = _mm256_permutevar8x32_epi32(
        x_exp, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0)
    );
    *e = _mm256_castsi256_si128(x_exp);
    return x;
}

//
// Calculates natural logarithm using AVX2 instructions
//
//   ln(x) = ln(x'*2^g), x' in [0,1), g in N
//         = ln(x') + g*ln(2)
//
// The logarithm in [0,1) is computed using the following Pade' form:
//
//   ln(1+x) = x - 0.5*x^2 + x^3*P(x)/Q(x)
//
__m256d nmc_mm256_log_pd(__m256d x) {
    __m256d x_orig = x;
    __m128i x_exp;

    // x := x', x_exp := g
    x = nmc_mm256_frexp_pd(x, &x_exp);

    // convert x_exp to packed double
    __m256d dx_exp = _mm256_cvtepi32_pd(x_exp);

    // blending
    __m256d lt_sqrth = _mm256_cmp_pd(
        x, _mm256_set1_pd(detail::dsqrth), 17 /* _CMP_LT_OQ */);

    // Adjust the argument and the exponent
    //       | 2*x - 1; e := e -1 , if x < sqrt(2)/2
    //  x := |
    //       | x - 1, otherwise

    // Precompute both branches
    // 2*x - 1
    __m256d x2m1 = _mm256_sub_pd(_mm256_add_pd(x, x), detail::nmc_m256d_one);

    // x - 1
    __m256d xm1 = _mm256_sub_pd(x, detail::nmc_m256d_one);

    // dx_exp - 1
    __m256d dx_exp_m1 = _mm256_sub_pd(dx_exp, detail::nmc_m256d_one);

    x = _mm256_blendv_pd(xm1, x2m1, lt_sqrth);
    dx_exp = _mm256_blendv_pd(dx_exp, dx_exp_m1, lt_sqrth);

    // compute P(x)
    __m256d px = _mm256_set1_pd(detail::P5log);
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P4log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P3log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P2log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P1log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::P0log));

    // xx := x^2
    // px := P(x)*x^3
    __m256d xx = _mm256_mul_pd(x, x);
    px = _mm256_mul_pd(px, x);
    px = _mm256_mul_pd(px, xx);

    // compute Q(x)
    __m256d qx = x;
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q4log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q3log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q2log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q1log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::Q0log));

    // x^3*P(x)/Q(x)
    __m256d ret = _mm256_div_pd(px, qx);

    // x^3*P(x)/Q(x) - g*ln(2)
    ret = _mm256_sub_pd(
        ret, _mm256_mul_pd(dx_exp, _mm256_set1_pd(detail::C3))
    );

    // -.5*x^ + x^3*P(x)/Q(x) - g*ln(2)
    ret = _mm256_sub_pd(ret, _mm256_mul_pd(_mm256_set1_pd(0.5), xx));

    // x -.5*x^ + x^3*P(x)/Q(x) - g*ln(2)
    ret = _mm256_add_pd(ret, x);

    // rounding error correction for ln(2)
    ret = _mm256_add_pd(ret, _mm256_mul_pd(dx_exp, _mm256_set1_pd(detail::C4)));

    // Treat exceptional cases
    __m256d is_inf = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_inf, 0 /* _CMP_EQ_OQ */);
    __m256d is_zero = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_zero, 0 /* _CMP_EQ_OQ */);
    __m256d is_neg = _mm256_cmp_pd(
        x_orig, detail::nmc_m256d_zero, 17 /* _CMP_LT_OQ */);
    __m256d is_denorm = nmc_mm256_subnormal_pd(x_orig);

    ret = _mm256_blendv_pd(ret, detail::nmc_m256d_inf, is_inf);
    ret = _mm256_blendv_pd(ret, detail::nmc_m256d_ninf, is_zero);

    // We treat denormalized cases as zeros
    ret = _mm256_blendv_pd(ret, detail::nmc_m256d_ninf, is_denorm);
    ret = _mm256_blendv_pd(ret, detail::nmc_m256d_nan, is_neg);
    return ret;
}

// Equivalent to exp(y*log(x))
__m256d nmc_mm256_pow_pd(__m256d x, __m256d y) {
    return nmc_mm256_exp_pd(_mm256_mul_pd(y, nmc_mm256_log_pd(x)));
}

} // end namespace multicore
} // end namespace mc
} // end namespace nest
