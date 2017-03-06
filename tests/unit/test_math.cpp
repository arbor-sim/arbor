#include <cmath>
#include <limits>

#include "../gtest.h"

#include <math.hpp>
#include <util/compat.hpp>

using namespace nest::mc::math;

TEST(math, pi) {
    // check regression against long double literal in implementation
    auto pi_ld = pi<long double>();
    auto pi_d = pi<double>();

    if (std::numeric_limits<long double>::digits>std::numeric_limits<double>::digits) {
        EXPECT_NE(0.0, pi_ld-pi_d);
    }
    else {
        EXPECT_EQ(0.0, pi_ld-pi_d);
    }

    // library quality of implementation dependent, but expect cos(pi) to be within
    // 1 epsilon of -1.

    auto eps_d = std::numeric_limits<double>::epsilon();
    auto cos_pi_d = std::cos(pi_d);
    EXPECT_LE(-1.0-eps_d, cos_pi_d);
    EXPECT_GE(-1.0+eps_d, cos_pi_d);

    auto eps_ld = std::numeric_limits<long double>::epsilon();
    auto cos_pi_ld = std::cos(pi_ld);
    EXPECT_LE(-1.0-eps_ld, cos_pi_ld);
    EXPECT_GE(-1.0+eps_ld, cos_pi_ld);
}

TEST(math, lerp) {
    // expect exact computation when u is zero or one
    double a = 1.0/3;
    double b = 11.0/7;

    EXPECT_EQ(a, lerp(a, b, 0.));
    EXPECT_EQ(b, lerp(a, b, 1.));

    // expect exact computation here as well
    EXPECT_EQ(2.75, lerp(2.0, 3.0, 0.75));

    // and otherwise to be close
    EXPECT_DOUBLE_EQ(100.101, lerp(100.1, 200.1, 0.00001));
    EXPECT_DOUBLE_EQ(200.099, lerp(100.1, 200.1, 0.99999));

    // should be able to lerp with differing types for end points and u
    EXPECT_EQ(0.25f, lerp(0.f, 1.f, 0.25));
}

TEST(math, frustrum) {
    // cross check against cone calculation
    auto cone_area = [](double l, double r) {
        return std::hypot(l,r)*r*pi<double>();
    };

    auto cone_volume = [](double l, double r) {
        return pi<double>()*square(r)*l/3.0;
    };

    EXPECT_DOUBLE_EQ(cone_area(5.0, 1.3), area_frustrum(5.0, 0.0, 1.3));
    EXPECT_DOUBLE_EQ(cone_volume(5.0, 1.3), volume_frustrum(5.0, 0.0, 1.3));

    double r1 = 7.0;
    double r2 = 9.0;
    double l = 11.0;

    double s = l*r2/(r2-r1);
    double ca = cone_area(s, r2)-cone_area(s-l, r1);
    double cv = cone_volume(s, r2)-cone_volume(s-l, r1);

    EXPECT_DOUBLE_EQ(ca, area_frustrum(l, r1, r2));
    EXPECT_DOUBLE_EQ(ca, area_frustrum(l, r2, r1));
    EXPECT_DOUBLE_EQ(cv, volume_frustrum(l, r1, r2));
    EXPECT_DOUBLE_EQ(cv, volume_frustrum(l, r2, r1));
}

TEST(math, infinity) {
    // check values for float, double, long double
    auto finf = infinity<float>();
    EXPECT_TRUE((std::is_same<float, decltype(finf)>::value));
    // COMPAT: use compatibility wrapper for isinf() thanks to xlC 13.1 bug.
    EXPECT_TRUE(compat::isinf(finf));
    EXPECT_GT(finf, 0.f);

    auto dinf = infinity<double>();
    EXPECT_TRUE((std::is_same<double, decltype(dinf)>::value));
    // COMPAT: use compatibility wrapper for isinf() thanks to xlC 13.1 bug.
    EXPECT_TRUE(compat::isinf(dinf));
    EXPECT_GT(dinf, 0.0);

    auto ldinf = infinity<long double>();
    EXPECT_TRUE((std::is_same<long double, decltype(ldinf)>::value));
    // COMPAT: use compatibility wrapper for isinf() thanks to xlC 13.1 bug.
    EXPECT_TRUE(compat::isinf(ldinf));
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
