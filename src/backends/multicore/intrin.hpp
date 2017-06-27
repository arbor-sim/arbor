//
// Custom transcendental intrinsics
//
// Implementation inspired by the VDT and Cephes libraries:
//    - https://github.com/dpiparo/vdt
//    - http://www.netlib.org/cephes/

#pragma once

#include <iostream>
#include <immintrin.h>

namespace nest {
namespace mc {
namespace multicore {

namespace detail {

const double exp_limit = 708;
const double px1exp = 1.26177193074810590878E-4;
const double px2exp = 3.02994407707441961300E-2;
const double px3exp = 9.99999999999999999910E-1;
const double px1log = 1.01875663804580931796e-4;
const double px2log = 4.97494994976747001425e-1;
const double px3log = 4.70579119878881725854e0;
const double px4log = 1.44989225341610930846e1;
const double px5log = 1.79368678507819816313e1;
const double px6log = 7.70838733755885391666E0;
const double qx1exp = 3.00198505138664455042E-6;
const double qx2exp = 2.52448340349684104192E-3;
const double qx3exp = 2.27265548208155028766E-1;
const double qx4exp = 2.00000000000000000009E0;
const double qx1log = 1.12873587189167450590e1;
const double qx2log = 4.52279145837532221105e1;
const double qx3log = 8.29875266912776603211e1;
const double qx4log = 7.11544750618563894466e1;
const double qx5log = 2.31251620126765340583E1;
const double log2e  = 1.4426950408889634073599; // 1/log(2)
const double c1d    = 6.93145751953125E-1;
const double c2d    = 1.42860682030941723212E-6;
const double c3d    = 2.121944400546905827679e-4;
const double c4d    = 0.693359375;
const uint64_t dmant_mask = ((1UL<<52) - 1) | (1UL << 63); // mantissa + sign
const uint64_t dexp_mask  = ((1UL<<11) - 1) << 52;
const int exp_bias = 1023;
const double dsqrth = 0.70710678118654752440;

// Useful constants in vector registers
const __m256d nmc_m256d_zero = _mm256_set1_pd(0.0);
const __m256d nmc_m256d_one  = _mm256_set1_pd(1.0);
const __m256d nmc_m256d_two  = _mm256_set1_pd(2.0);
const __m256d nmc_m256d_nan  = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
const __m256d nmc_m256d_inf  = _mm256_set1_pd(std::numeric_limits<double>::infinity());
const __m256d nmc_m256d_ninf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
}

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

__m256d nmc_mm256_exp_pd(__m256d x) {
    __m256d x_orig = x;
    __m256d px = _mm256_floor_pd(
        _mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(detail::log2e), x),
            _mm256_set1_pd(0.5)
        )
    );

    __m128i n = _mm256_cvtpd_epi32(px);

    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(detail::c1d)));
    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(detail::c2d)));

    __m256d xx = _mm256_mul_pd(x, x);
    px = _mm256_set1_pd(detail::px1exp);
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px2exp));
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px3exp));
    px = _mm256_mul_pd(px, x);

    __m256d qx = _mm256_set1_pd(detail::qx1exp);
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx2exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx3exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx4exp));

    x = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
    x = _mm256_add_pd(detail::nmc_m256d_one,
                      _mm256_mul_pd(detail::nmc_m256d_two, x));


    // x *= 2**n
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

__m256d nmc_mm256_log_pd(__m256d x) {
    __m256d x_orig = x;
    __m128i x_exp;
    x = nmc_mm256_frexp_pd(x, &x_exp);

    // convert x_exp to packed double
    __m256d dx_exp = _mm256_cvtepi32_pd(x_exp);

    // blending
    __m256d lt_sqrth = _mm256_cmp_pd(
        x, _mm256_set1_pd(detail::dsqrth), 17 /* _CMP_LT_OQ */);

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
    __m256d px = _mm256_set1_pd(detail::px1log);
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px2log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px3log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px4log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px5log));
    px = _mm256_mul_pd(px, x);
    px = _mm256_add_pd(px, _mm256_set1_pd(detail::px6log));

    // x^2
    __m256d xx = _mm256_mul_pd(x, x);
    px = _mm256_mul_pd(px, x);
    px = _mm256_mul_pd(px, xx);

    // compute Q(x)
    __m256d qx = x;
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx1log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx2log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx3log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx4log));
    qx = _mm256_mul_pd(qx, x);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(detail::qx5log));

    __m256d ret = _mm256_div_pd(px, qx);
    ret = _mm256_sub_pd(
        ret, _mm256_mul_pd(dx_exp, _mm256_set1_pd(detail::c3d))
    );
    ret = _mm256_sub_pd(ret, _mm256_mul_pd(_mm256_set1_pd(0.5), xx));
    ret = _mm256_add_pd(ret, x);
    ret = _mm256_add_pd(ret, _mm256_mul_pd(dx_exp, _mm256_set1_pd(detail::c4d)));

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
