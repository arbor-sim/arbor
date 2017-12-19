#include <cmath>
#include <limits>

#include "../gtest.h"

#include <math.hpp>
#include <util/compat.hpp>

using namespace arb::math;

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

TEST(math, signum) {
    EXPECT_EQ(1, signum(1));
    EXPECT_EQ(1, signum(2));
    EXPECT_EQ(1, signum(3.f));
    EXPECT_EQ(1, signum(4.));

    EXPECT_EQ(0, signum(0));
    EXPECT_EQ(0, signum(0.f));
    EXPECT_EQ(0, signum(0.));

    EXPECT_EQ(-1, signum(-1));
    EXPECT_EQ(-1, signum(-2));
    EXPECT_EQ(-1, signum(-3.f));
    EXPECT_EQ(-1, signum(-4.));

    double denorm = 1e-308;
    EXPECT_EQ(1, signum(denorm));
    EXPECT_EQ(-1, signum(-denorm));

    double negzero = std::copysign(0., -1.);
    EXPECT_EQ(0, signum(negzero));

    EXPECT_EQ(1, signum(infinity<double>()));
    EXPECT_EQ(1, signum(infinity<float>()));
    EXPECT_EQ(-1, signum(-infinity<double>()));
    EXPECT_EQ(-1, signum(-infinity<float>()));
}


TEST(quaternion, ctor) {
    // scalar
    quaternion q1(3.5);

    EXPECT_EQ(3.5, q1.w);
    EXPECT_EQ(0., q1.x);
    EXPECT_EQ(0., q1.y);
    EXPECT_EQ(0., q1.z);

    // pure imaginery
    quaternion q2(1.5, 2.5, 3.5);

    EXPECT_EQ(0, q2.w);
    EXPECT_EQ(1.5, q2.x);
    EXPECT_EQ(2.5, q2.y);
    EXPECT_EQ(3.5, q2.z);

    // all components
    quaternion q3(0.5, 1.5, 2.5, 3.5);

    EXPECT_EQ(0.5, q3.w);
    EXPECT_EQ(1.5, q3.x);
    EXPECT_EQ(2.5, q3.y);
    EXPECT_EQ(3.5, q3.z);

    // copy ctor
    quaternion q4(q3);

    EXPECT_EQ(0.5, q4.w);
    EXPECT_EQ(1.5, q4.x);
    EXPECT_EQ(2.5, q4.y);
    EXPECT_EQ(3.5, q4.z);
}

TEST(quaternion, assign) {
    quaternion q1(0.5, 1.5, 2.5, 3.5);
    quaternion q2(7.3, -2.4, 11.1, -9);

    q2 = q1;

    EXPECT_EQ(0.5, q2.w);
    EXPECT_EQ(1.5, q2.x);
    EXPECT_EQ(2.5, q2.y);
    EXPECT_EQ(3.5, q2.z);

    q2 = -11.5;

    EXPECT_EQ(-11.5, q2.w);
    EXPECT_EQ(0, q2.x);
    EXPECT_EQ(0, q2.y);
    EXPECT_EQ(0, q2.z);
}

TEST(quaternion, equality) {
    quaternion q1(1., 2., 3., 5.5);

    quaternion q2(q1);
    EXPECT_EQ(q1, q2);

    q2 = q1;
    q2.w += 0.5;
    EXPECT_NE(q1, q2);

    q2 = q1;
    q2.x += 0.5;
    EXPECT_NE(q1, q2);

    q2 = q1;
    q2.y += 0.5;
    EXPECT_NE(q1, q2);

    q2 = q1;
    q2.z += 0.5;
    EXPECT_NE(q1, q2);
}

TEST(quaternion, unaryop) {
    quaternion q(2, -3, 4.5, -5);

    EXPECT_EQ(quaternion(-2, 3, -4.5, 5), -q);
    EXPECT_EQ(quaternion(2, 3, -4.5, 5), q.conj());

    quaternion r(10, 6, 4, 37);
    EXPECT_EQ(1521., r.sqnorm());
    EXPECT_DOUBLE_EQ(39., r.norm());
}

TEST(quaternion, binop) {
    quaternion q1(2, -3, 4.5, -5);
    quaternion q2(0.5, 1.5, -2.5, 3.5);

    EXPECT_EQ(quaternion(2.5, -1.5, 2, -1.5), q1+q2);
    EXPECT_EQ(quaternion(1.5, -4.5, 7, -8.5), q1-q2);
    EXPECT_EQ(quaternion(34.25, 4.75, 0.25, 5.25), q1*q2);
    EXPECT_EQ(quaternion(42, -41.5, 71, -131), q1^q2);
}

TEST(quaternion, assignop) {
    quaternion q1(2, -3, 4.5, -5);
    quaternion q2(0.5, 1.5, -2.5, 3.5);

    quaternion q;
    q = q1;
    q += q2;
    EXPECT_EQ(q1+q2, q);

    q = q1;
    q -= q2;
    EXPECT_EQ(q1-q2, q);

    q = q1;
    q *= q2;
    EXPECT_EQ(q1*q2, q);

    q = q1;
    q ^= q2;
    EXPECT_EQ(q1^q2, q);
}

TEST(quaternion, rotate) {
    double deg_to_rad = pi<double>()/180.;
    double sqrt3o2 = std::sqrt(3.)/2.;
    double eps = 1e-15;

    quaternion q(0, 1, 2, 3);

    auto r = q^rotation_x(deg_to_rad*30);
    EXPECT_NEAR(0., r.w, eps);
    EXPECT_NEAR(q.x, r.x, eps);
    EXPECT_NEAR(q.y*sqrt3o2-q.z/2., r.y, eps);
    EXPECT_NEAR(q.y/2.+q.z*sqrt3o2, r.z, eps);

    r = q^rotation_y(deg_to_rad*30);
    EXPECT_NEAR(0., r.w, eps);
    EXPECT_NEAR(q.x*sqrt3o2+q.z/2., r.x, eps);
    EXPECT_NEAR(q.y, r.y, eps);
    EXPECT_NEAR(-q.x/2.+q.z*sqrt3o2, r.z, eps);

    r = q^rotation_z(deg_to_rad*30);
    EXPECT_NEAR(0., r.w, eps);
    EXPECT_NEAR(q.x*sqrt3o2-q.y/2., r.x, eps);
    EXPECT_NEAR(q.x/2.+q.y*sqrt3o2, r.y, eps);
    EXPECT_NEAR(q.z, r.z, eps);
}

TEST(math, exprelr) {
    constexpr double dmin = std::numeric_limits<double>::min();
    constexpr double dmax = std::numeric_limits<double>::max();
    constexpr double deps = std::numeric_limits<double>::epsilon();
    double inputs[] = {-1.,  -0.,  0.,  1., -dmax,  -dmin,  dmin,  dmax, -deps, deps, 10*deps, 100*deps, 1000*deps};

    for (auto x: inputs) {
        if (std::fabs(x)<deps) EXPECT_EQ(1.0, exprelr(x));
        else                   EXPECT_EQ(x/std::expm1(x), exprelr(x));
    }
}

TEST(math, minmax) {
    constexpr double inf = std::numeric_limits<double>::infinity();

    struct X {
        double lhs;
        double rhs;
        double expected_min;
        double expected_max;
    };

    std::vector<X> inputs = {
        {  0,    1,    0,   1},
        { -1,    1,   -1,   1},
        { 42,   42,   42,  42},
        {inf, -inf, -inf, inf},
        {  0,  inf,    0, inf},
        {  0, -inf, -inf,   0},
    };

    for (auto x: inputs) {
        // Call min and max with arguments in both possible orders.
        EXPECT_EQ(min(x.lhs, x.rhs), x.expected_min);
        EXPECT_EQ(min(x.rhs, x.lhs), x.expected_min);
        EXPECT_EQ(max(x.lhs, x.rhs), x.expected_max);
        EXPECT_EQ(max(x.rhs, x.lhs), x.expected_max);
    }
}

