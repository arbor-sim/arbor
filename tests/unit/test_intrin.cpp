#include <cmath>
#include <limits>
#include <backends/multicore/intrin.hpp>
#include <immintrin.h>

#include "../gtest.h"
//#include "../../../vdt/include/exp.h"

using namespace nest::mc::multicore;

constexpr double dqnan = std::numeric_limits<double>::quiet_NaN();
constexpr double dmax = std::numeric_limits<double>::max();
constexpr double dmin = std::numeric_limits<double>::min();
constexpr double dmin_denorm = std::numeric_limits<double>::denorm_min();
constexpr double dinf = std::numeric_limits<double>::infinity();

constexpr double values[] = {
        -3, -2, -1, 0,
         1,  2,  3, 10,
        dqnan, -dinf, dinf, 0,
        dmin, dmax, dmin_denorm, 0
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
        __m256d vv = nmc_mm256_exp_pd(vvalues[i]);
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
