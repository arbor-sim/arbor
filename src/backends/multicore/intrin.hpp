//
// Custom transcendental intrinsics
//
// Implementation inspired by the VDT library:
//    - https://github.com/dpiparo/vdt
//

#pragma once

#include <iostream>
#include <immintrin.h>

namespace nest {
namespace mc {
namespace multicore {

namespace details {
const double exp_limit = 708;
const double px1exp = 1.26177193074810590878E-4;
const double px2exp = 3.02994407707441961300E-2;
const double px3exp = 9.99999999999999999910E-1;
const double qx1exp = 3.00198505138664455042E-6;
const double qx2exp = 2.52448340349684104192E-3;
const double qx3exp = 2.27265548208155028766E-1;
const double qx4exp = 2.00000000000000000009E0;
const double log2e  = 1.4426950408889634073599; // 1/log(2)
const double c1d    = 6.93145751953125E-1;
const double c2d    = 1.42860682030941723212E-6;

__m256d nmc_mm256_one = _mm256_set1_pd(1.0);
__m256d nmc_mm256_two = _mm256_set1_pd(2.0);
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

__m256d nmc_mm256_exp_pd(__m256d x) {
    __m256d x_orig = x;
    __m256d px = _mm256_floor_pd(
        _mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(details::log2e), x),
            _mm256_set1_pd(0.5)
        )
    );

    __m128i n = _mm256_cvtpd_epi32(px);

    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(details::c1d)));
    x = _mm256_sub_pd(x, _mm256_mul_pd(px, _mm256_set1_pd(details::c2d)));

    __m256d xx = _mm256_mul_pd(x, x);
    px = _mm256_set1_pd(details::px1exp);
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(details::px2exp));
    px = _mm256_mul_pd(px, xx);
    px = _mm256_add_pd(px, _mm256_set1_pd(details::px3exp));
    px = _mm256_mul_pd(px, x);

    __m256d qx = _mm256_set1_pd(details::qx1exp);
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(details::qx2exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(details::qx3exp));
    qx = _mm256_mul_pd(qx, xx);
    qx = _mm256_add_pd(qx, _mm256_set1_pd(details::qx4exp));

    x = _mm256_div_pd(px, _mm256_sub_pd(qx, px));
    x = _mm256_add_pd(details::nmc_mm256_one,
                      _mm256_mul_pd(details::nmc_mm256_two, x));


    // x *= 2**n
    __m256i n64 = _mm256_cvtepi32_epi64(n);
    n64 = _mm256_add_epi64(n64, _mm256_set1_epi64x(1023));
    n64 = _mm256_sll_epi64(n64, _mm_set_epi64x(0, 52));
    x = _mm256_mul_pd(x, _mm256_castsi256_pd(n64));

    // Treat exceptional cases
    __m256d is_large = _mm256_cmp_pd(
        x_orig, _mm256_set1_pd(details::exp_limit), 30 /* _CMP_GT_OQ */
    );
    __m256d is_small = _mm256_cmp_pd(
        x_orig, _mm256_set1_pd(-details::exp_limit), 17 /* _CMP_LT_OQ */
    );
    __m256d is_nan = _mm256_cmp_pd(x_orig, x_orig, 3 /* _CMP_UNORD_Q */ );

    __m256d nan = _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN());
    __m256d inf  = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d zero = _mm256_set1_pd(0);
    x = _mm256_blendv_pd(x, inf, is_large);
    x = _mm256_blendv_pd(x, zero, is_small);
    x = _mm256_blendv_pd(x, nan, is_nan);
    return x;

}

} // end namespace multicore
} // end namespace mc
} // end namespace nest
