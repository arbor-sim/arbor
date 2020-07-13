#include <cmath>
#include <vector>

#include <arbor/math.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "util/piecewise.hpp"

#include "../test/gtest.h"
#include "common.hpp"
#include "common_cells.hpp"

using namespace arb;

namespace {

struct v3 {
    double x, y, z;
    friend double dot(v3 p, v3 q) {
        return p.x*q.x+p.y*q.y+p.z*q.z;
    }
    friend v3 cross(v3 p, v3 q) {
        return {p.y*q.z-p.z*q.y, p.z*q.x-p.x*q.z, p.x*q.y-p.y*q.x};
    }
    friend v3 operator*(v3 p, double s) {
        return {s*p.x, s*p.y, s*p.z};
    }
    friend v3 operator*(double s, v3 p) {
        return p*s;
    }
    friend v3 operator/(v3 p, double s) {
        return p*(1./s);
    }
    friend v3 operator+(v3 p, v3 q) {
        return {p.x+q.x, p.y+q.y, p.z+q.z};
    }
    friend double length(v3 x) {
        return std::sqrt(dot(x, x));
    }
};

::testing::AssertionResult mpoint_almost_eq(mpoint a, mpoint b) {
    using FP = testing::internal::FloatingPoint<double>;
    if (FP(a.x).AlmostEquals(FP(b.x)) &&
        FP(a.y).AlmostEquals(FP(b.y)) &&
        FP(a.z).AlmostEquals(FP(b.z)) &&
        FP(a.radius).AlmostEquals(FP(b.radius)))
    {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure() << "mpoint values "
            << '(' << a.x << ',' << a.y << ',' << a.z << ';' << a.radius << ')'
            << " and "
            << '(' << b.x << ',' << b.y << ',' << b.z << ';' << b.radius << ')'
            << " differ";
    }
}

::testing::AssertionResult v3_almost_eq(v3 a, v3 b) {
    using FP = testing::internal::FloatingPoint<double>;
    if (FP(a.x).AlmostEquals(FP(b.x)) &&
        FP(a.y).AlmostEquals(FP(b.y)) &&
        FP(a.z).AlmostEquals(FP(b.z)))
    {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure() << "xyz values "
            << '(' << a.x << ',' << a.y << ',' << a.z << ')'
            << " and "
            << '(' << b.x << ',' << b.y << ',' << b.z << ')'
            << " differ";
    }
}

} // anonymous namespace

TEST(isometry, translate) {
    mpoint p{1, 2, 3, 4};

    {
        isometry id;
        mpoint q = id.apply(p);
        EXPECT_EQ(p.x, q.x);
        EXPECT_EQ(p.y, q.y);
        EXPECT_EQ(p.z, q.z);
        EXPECT_EQ(p.radius, q.radius);
    }
    {
        isometry shift = isometry::translate(0.5, 1.5, 3.5);
        mpoint q = shift.apply(p);

        EXPECT_EQ(p.x+0.5, q.x);
        EXPECT_EQ(p.y+1.5, q.y);
        EXPECT_EQ(p.z+3.5, q.z);
        EXPECT_EQ(p.radius, q.radius);

        // Should work with anything with floating point x, y, z.
        struct v3 {
            double x, y, z;
        };

        isometry shift_bis = isometry::translate(v3{0.5, 1.5, 3.5});
        mpoint q_bis = shift_bis.apply(p);
        EXPECT_EQ(q, q_bis);
    }
}

TEST(isometry, rotate) {
    {
        // Rotation about axis through p should do nothing.
        mpoint p{1, 2, 3, 4};

        isometry rot = isometry::rotate(1.23, p);
        mpoint q = rot.apply(p);

        EXPECT_TRUE(mpoint_almost_eq(p, q));
    }
    {
        // Rotate about an arbitrary axis.
        //
        // Geometry says rotating x about (normalized) u by
        // θ should give a vector u(u.x) + (u×x)×u cos θ + u×x sin 0

        double theta = 0.234;
        v3 axis{-1, 3, 2.2};

        isometry rot = isometry::rotate(theta, axis);
        v3 x = {1, 2, 3};
        v3 q = rot.apply(x);

        v3 u = axis/length(axis);
        v3 p = u*dot(u, x) + cross(cross(u, x), u)*std::cos(theta) + cross(u, x)*std::sin(theta);

        EXPECT_TRUE(v3_almost_eq(p, q));
    }
}

TEST(isometry, compose) {
    // For an isometry (r, t) (r the rotation, t the translation),
    // we compose them by:
    //   (r, t) ∘ (s, u) = (sr, t+u).
    //
    // On the other hand, for a vector x:
    //   (r, t) ((s, u) x)
    //      = (r, t) (sx + u)
    //      = rsx + ru + t
    //      = (rs, ru+t) x
    //      = ((s, ru) ∘ (r, t)) x

    double theta1 = -0.3;
    v3 axis1{1, -3, 4};
    v3 shift1{-2, -1, 3};

    double theta2 = 0.6;
    v3 axis2{-1, 2, 0};
    v3 shift2{3, 0, -1};

    v3 p{5., 6., 7.};

    // Construct (r, t):
    isometry r = isometry::rotate(theta1, axis1);
    isometry r_t = r*isometry::translate(shift1);

    // Construct (s, u):
    isometry s = isometry::rotate(theta2, axis2);
    isometry s_u = s*isometry::translate(shift2);

    // Construct (s, ru):
    isometry s_ru = s*isometry::translate(r.apply(shift2));

    // Compare application of s_u and r_t against application
    // of (s, ru) ∘ (r, t).

    v3 u = r_t.apply(s_u.apply(p));
    v3 v = (s_ru * r_t).apply(p);

    EXPECT_TRUE(v3_almost_eq(u, v));
}

TEST(place_pwlin, cable) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<mpoint>;

    // L-shaped simple cable.
    // 0.25 of the length in the z-direction,
    // 0.75 of the length in the x-direction.

    pvec parents = {mnpos, 0, 1};
    svec points = {
        {0,  0,  0,  2},
        {0,  0,  1,  2},
        {3,  0,  1,  2}
    };

    auto sm = segments_from_points(points, parents);
    morphology m(sm);

    {
        // With no transformation:
        place_pwlin pl(m);

        // Interpolated points.
        mpoint p_0 = pl.at(mlocation{0, 0.});
        mpoint p_1 = pl.at(mlocation{0, 0.125});
        mpoint p_2 = pl.at(mlocation{0, 0.25});
        mpoint p_3 = pl.at(mlocation{0, 0.5});

        // Expected results.
        mpoint x_0{0.0, 0.0, 0.0, 2.};
        mpoint x_1{0.0, 0.0, 0.5, 2.};
        mpoint x_2{0.0, 0.0, 1.0, 2.};
        mpoint x_3{1.0, 0.0, 1.0, 2.};

        EXPECT_TRUE(mpoint_almost_eq(x_0, p_0));
        EXPECT_TRUE(mpoint_almost_eq(x_1, p_1));
        EXPECT_TRUE(mpoint_almost_eq(x_2, p_2));
        EXPECT_TRUE(mpoint_almost_eq(x_3, p_3));
    }
    {
        // With a rotation and translation:

        double theta = -0.3;
        v3 axis{1, -3, 4};
        v3 shift{-2, -1, 3};

        isometry iso = isometry::rotate(theta, axis)*isometry::translate(shift);
        place_pwlin pl(m, iso);

        // Interpolated points.
        mpoint p_0 = pl.at(mlocation{0, 0.});
        mpoint p_1 = pl.at(mlocation{0, 0.125});
        mpoint p_2 = pl.at(mlocation{0, 0.25});
        mpoint p_3 = pl.at(mlocation{0, 0.5});

        // Expected results.
        mpoint x_0 = iso.apply(mpoint{0.0, 0.0, 0.0, 2.});
        mpoint x_1 = iso.apply(mpoint{0.0, 0.0, 0.5, 2.});
        mpoint x_2 = iso.apply(mpoint{0.0, 0.0, 1.0, 2.});
        mpoint x_3 = iso.apply(mpoint{1.0, 0.0, 1.0, 2.});

        EXPECT_TRUE(mpoint_almost_eq(x_0, p_0));
        EXPECT_TRUE(mpoint_almost_eq(x_1, p_1));
        EXPECT_TRUE(mpoint_almost_eq(x_2, p_2));
        EXPECT_TRUE(mpoint_almost_eq(x_3, p_3));
    }
}

TEST(place_pwlin, branched) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<mpoint>;

    // Y-shaped branched morphology.
    // Second branch (branch 1) tapers radius.

    pvec parents = {mnpos, 0, 1, 2, 3, 4, 2, 6};
    svec points = {
        // branch 0
        { 0,  0,  0,  2},
        { 0,  0,  1,  2},
        { 3,  0,  1,  2},
        // branch 1
        { 3,  0,  1,  1.0},
        { 3,  1,  1,  1.0},
        { 3,  2,  1,  0.0},
        // branch 2
        { 3,  0,  1,  2},
        { 3,  -1, 1,  2}
    };

    auto sm = segments_from_points(points, parents);
    morphology m(sm);

    isometry iso = isometry::translate(2, 3, 4);
    place_pwlin pl(m, iso);

    // Examnine points on branch 1.

    mpoint p_0 = pl.at(mlocation{1, 0.});
    mpoint p_1 = pl.at(mlocation{1, 0.25});
    mpoint p_2 = pl.at(mlocation{1, 0.75});

    mpoint x_0 = iso.apply(mpoint{3, 0,   1, 1});
    mpoint x_1 = iso.apply(mpoint{3, 0.5, 1, 1});
    mpoint x_2 = iso.apply(mpoint{3, 1.5, 1, 0.5});

    EXPECT_TRUE(mpoint_almost_eq(x_0, p_0));
    EXPECT_TRUE(mpoint_almost_eq(x_1, p_1));
    EXPECT_TRUE(mpoint_almost_eq(x_2, p_2));
}
