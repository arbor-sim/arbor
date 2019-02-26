#include <cmath>
#include <limits>
#include <backends/multicore/intrin.hpp>
#include <immintrin.h>

#include <util/span.hpp>

#include "../gtest.h"

using namespace arb::multicore;

using arb::util::make_span;
using arb::util::size;

constexpr double dqnan = std::numeric_limits<double>::quiet_NaN();
constexpr double dmax = std::numeric_limits<double>::max();
constexpr double dmin = std::numeric_limits<double>::min();
constexpr double dmin_denorm = std::numeric_limits<double>::denorm_min();
constexpr double dinf = std::numeric_limits<double>::infinity();
constexpr double deps = std::numeric_limits<double>::epsilon();

constexpr double values[] = {
    -300, -3, -2, -1,
    1,  2,  3, 600,
    dqnan, -dinf, dinf, -0.0,
    dmin, dmax, dmin_denorm, +0.0,
};

TEST(intrin, exp256) {
    constexpr size_t simd_len = 4;

    __m256d vvalues[] = {
        _mm256_set_pd(values[3],  values[2],  values[1],  values[0]),
        _mm256_set_pd(values[7],  values[6],  values[5],  values[4]),
        _mm256_set_pd(values[11], values[10], values[9],  values[8]),
        _mm256_set_pd(values[15], values[14], values[13], values[12])
    };


    for (size_t i = 0; i < 16/simd_len; ++i) {
        __m256d vv = arb_mm256_exp_pd(vvalues[i]);
        double *intrin = (double *) &vv;
        for (size_t j = 0; j < simd_len; ++j) {
            double v = values[i*simd_len + j];
            if (std::isnan(v)) {
                EXPECT_TRUE(std::isnan(intrin[j]));
            }
            else {
                EXPECT_DOUBLE_EQ(std::exp(v), intrin[j]);
            }
        }
    }
}

TEST(intrin, abs256) {
    constexpr size_t simd_len = 4;

    __m256d vvalues[] = {
        _mm256_set_pd(-42.,  -0.,  0.,  42.),
        _mm256_set_pd(-dmin,  -dmax,  -dmax, dqnan),
    };


    for (auto i: {0u, 1u}) {
        auto x = arb_mm256_abs_pd(vvalues[i]);
        double* in  = (double*) &(vvalues[i]);
        double* out = (double*) &x;
        for (size_t j = 0; j < simd_len; ++j) {
            double v = in[j];
            if (std::isnan(v)) {
                EXPECT_TRUE(std::isnan(out[j]));
            }
            else {
                EXPECT_DOUBLE_EQ(std::fabs(v), out[j]);
            }
        }
    }
}

TEST(intrin, min256) {
    constexpr size_t simd_len = 4;

    __m256d lhs = _mm256_set_pd(-2,  2,  -dinf,  dinf);
    __m256d rhs = _mm256_set_pd(2,  -2,  42, 1);

    auto mi = arb_mm256_min_pd(lhs, rhs);

    double* lp  = (double*) &lhs;
    double* rp  = (double*) &rhs;
    double* mip = (double*) &mi;
    for (size_t j = 0; j < simd_len; ++j) {
        EXPECT_DOUBLE_EQ(std::min(lp[j], rp[j]), mip[j]);
    }
}

TEST(intrin, exprelr256) {
    constexpr size_t simd_len = 4;

    // TODO: the third set of test values is commented out because it currently
    // fails, because our implementation uses x/(exp(x)-1), when we should use
    // x/expm(1) to handle rounding errors when x is close to 0.
    // This test can be added once we have an implementation of expm1.
    __m256d vvalues[] = {
        _mm256_set_pd(-1.,  -0.,  0.,  1.),
        _mm256_set_pd(-dmax,  -dmin,  dmin,  dmax),
        //_mm256_set_pd(-deps, deps, 10*deps,  100*deps),
    };

    for (auto i: make_span(0, size(vvalues))) {
        auto x = arb_mm256_exprelr_pd(vvalues[i]);
        double* in  = (double*) &(vvalues[i]);
        double* out = (double*) &x;
        for (size_t j = 0; j < simd_len; ++j) {
            double v = in[j];
            if (std::fabs(v)<deps) {
                EXPECT_DOUBLE_EQ(1.0, out[j]);
            }
            else {
                EXPECT_DOUBLE_EQ(v/expm1(v), out[j]);
            }
        }
    }
}

TEST(intrin, frexp256) {
    constexpr size_t simd_len = 4;

    __m256d vvalues[] = {
        _mm256_set_pd(values[3],  values[2],  values[1],  values[0]),
        _mm256_set_pd(values[7],  values[6],  values[5],  values[4]),
        _mm256_set_pd(values[11], values[10], values[9],  values[8]),
        _mm256_set_pd(values[15], values[14], values[13], values[12])
    };


    for (size_t i = 0; i < 16/simd_len; ++i) {
        __m128i vexp;
        __m256d vbase = arb_mm256_frexp_pd(vvalues[i], &vexp);
        double *vbase_ = (double *) &vbase;
        int    *vexp_  = (int *) &vexp;
        for (size_t j = 0; j < simd_len; ++j) {
            double v = values[i*simd_len + j];
            if (std::fpclassify(v) == FP_SUBNORMAL) {
                // FIXME: our implementation treats subnormals as zeros
                v = 0;
            }

            int exp;
            double base = std::frexp(v, &exp);
            if (std::isnan(v)) {
                EXPECT_TRUE(std::isnan(vbase_[j]));
            }
            else if (std::isinf(v)) {
                // Returned exponents are implementation defined in this case
                EXPECT_DOUBLE_EQ(base, vbase_[j]);
            }
            else {
                EXPECT_DOUBLE_EQ(base, vbase_[j]);
                EXPECT_EQ(exp, vexp_[j]);
            }
        }
    }
}


TEST(intrin, log256) {
    constexpr size_t simd_len = 4;

    __m256d vvalues[] = {
        _mm256_set_pd(values[3],  values[2],  values[1],  values[0]),
        _mm256_set_pd(values[7],  values[6],  values[5],  values[4]),
        _mm256_set_pd(values[11], values[10], values[9],  values[8]),
        _mm256_set_pd(values[15], values[14], values[13], values[12])
    };


    for (size_t i = 0; i < 16/simd_len; ++i) {
        __m256d vv = arb_mm256_log_pd(vvalues[i]);
        double *intrin = (double *) &vv;
        for (size_t j = 0; j < simd_len; ++j) {
            double v = values[i*simd_len + j];
            if (std::fpclassify(v) == FP_SUBNORMAL) {
                // FIXME: our implementation treats subnormals as zeros
                v = 0;
            }
            if (v < 0 || std::isnan(v)) {
                EXPECT_TRUE(std::isnan(intrin[j]));
            }
            else {
                EXPECT_DOUBLE_EQ(std::log(v), intrin[j]);
            }
        }
    }
}
