#include <cmath>
#include <limits>

#include "gtest.h"
#include <math.hpp>

using namespace nest::mc::math;

TEST(math, infinity) {
    // check values for float, double, long double
    auto finf = infinity<float>();
    EXPECT_TRUE((std::is_same<float, decltype(finf)>::value));
    EXPECT_TRUE(std::isinf(finf));
    EXPECT_GT(finf, 0.f);

    auto dinf = infinity<double>();
    EXPECT_TRUE((std::is_same<double, decltype(dinf)>::value));
    EXPECT_TRUE(std::isinf(dinf));
    EXPECT_GT(dinf, 0.0);

    auto ldinf = infinity<long double>();
    EXPECT_TRUE((std::is_same<long double, decltype(ldinf)>::value));
    EXPECT_TRUE(std::isinf(ldinf));
    EXPECT_GT(ldinf, 0.0l);

    // check default value promotes correctly (i.e., acts like INFINITY)
    struct {
        float f;
        double d;
        long double ld;
    } check = {infinity<>(), infinity<>(), infinity<>()};

    EXPECT_EQ(std::numeric_limits<float>::infinity(), check.f);
    EXPECT_EQ(std::numeric_limits<double>::infinity(), check.d);
    EXPECT_EQ(std::numeric_limits<long double>::infinity(), check.ld);
}
