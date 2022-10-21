#include <cmath>
#include <vector>

#include <arbor/math.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"

#include <gtest/gtest.h>
#include "../common_cells.hpp"

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

TEST(place_pwlin, all_at) {
    // One branch, two discontinguous segments.
    {
        mpoint p0p{0, 0, 0, 1};
        mpoint p0d{2, 3, 4, 5};
        mpoint p1p{3, 4, 5, 7};
        mpoint p1d{6, 6, 7, 8};

        segment_tree tree;
        msize_t s0 = tree.append(mnpos, p0p, p0d, 0);
        msize_t s1 = tree.append(s0, p1p, p1d, 0);

        morphology m(tree);
        mprovider p(m, label_dict{});
        place_pwlin place(m);

        mlocation_list end_s0 = thingify(ls::most_distal(reg::segment(s0)), p);
        ASSERT_EQ(1u, end_s0.size());
        mlocation_list end_s1 = thingify(ls::most_distal(reg::segment(s1)), p);
        ASSERT_EQ(1u, end_s1.size());

        auto points_end_s1 = place.all_at(end_s1[0]);
        ASSERT_EQ(1u, points_end_s1.size());
        EXPECT_TRUE(mpoint_almost_eq(p1d, points_end_s1[0]));

        auto points_end_s0 = place.all_at(end_s0[0]);
        ASSERT_EQ(2u, points_end_s0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_end_s0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p1p, points_end_s0[1]));
    }

    // One branch, multiple zero-length segments at end.
    {
        mpoint p0p{0, 0, 0, 1};
        mpoint p0d{2, 3, 4, 5};
        mpoint p1p{3, 4, 5, 7};
        mpoint p1d = p1p;
        mpoint p2p{6, 6, 7, 8};
        mpoint p2d = p2p;

        segment_tree tree;
        msize_t s0 = tree.append(mnpos, p0p, p0d, 0);
        msize_t s1 = tree.append(s0, p1p, p1d, 0);
        (void)tree.append(s1, p2p, p2d, 0);

        morphology m(tree);
        mprovider p(m, label_dict{});
        place_pwlin place(m);

        auto points_end_b0 = place.all_at(mlocation{0, 1});
        ASSERT_EQ(3u, points_end_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_end_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p1d, points_end_b0[1]));
        EXPECT_TRUE(mpoint_almost_eq(p2d, points_end_b0[2]));
    }

    // Zero length branch comprising multiple zero-length segments.
    // (Please don't do this.)
    {
        mpoint p0p{2, 3, 4, 5};
        mpoint p0d = p0p;
        mpoint p1p{3, 4, 5, 7};
        mpoint p1d = p1p;
        mpoint p2p{6, 6, 7, 8};
        mpoint p2d = p2p;

        segment_tree tree;
        msize_t s0 = tree.append(mnpos, p0p, p0d, 0);
        msize_t s1 = tree.append(s0, p1p, p1d, 0);
        (void)tree.append(s1, p2p, p2d, 0);

        morphology m(tree);
        mprovider p(m, label_dict{});
        place_pwlin place(m);

        auto points_begin_b0 = place.all_at(mlocation{0, 0});
        ASSERT_EQ(3u, points_begin_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_begin_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p1d, points_begin_b0[1]));
        EXPECT_TRUE(mpoint_almost_eq(p2d, points_begin_b0[2]));

        auto points_mid_b0 = place.all_at(mlocation{0, 0.5});
        ASSERT_EQ(3u, points_mid_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_mid_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p1d, points_mid_b0[1]));
        EXPECT_TRUE(mpoint_almost_eq(p2d, points_mid_b0[2]));

        auto points_end_b0 = place.all_at(mlocation{0, 1});
        ASSERT_EQ(3u, points_end_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_end_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p1d, points_end_b0[1]));
        EXPECT_TRUE(mpoint_almost_eq(p2d, points_end_b0[2]));
    }

    // Zero length branch comprising single zero-length segment with differing radius.
    // (Please don't do this either.)
    {
        mpoint p0p{2, 3, 4, 5};
        mpoint p0d{2, 3, 4, 8};

        segment_tree tree;
        (void)tree.append(mnpos, p0p, p0d, 0);

        morphology m(tree);
        mprovider p(m, label_dict{});
        place_pwlin place(m);

        auto points_begin_b0 = place.all_at(mlocation{0, 0});
        ASSERT_EQ(2u, points_begin_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0p, points_begin_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_begin_b0[1]));

        auto points_mid_b0 = place.all_at(mlocation{0, 0.5});
        ASSERT_EQ(2u, points_begin_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0p, points_begin_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_begin_b0[1]));

        auto points_end_b0 = place.all_at(mlocation{0, 1});
        ASSERT_EQ(2u, points_begin_b0.size());
        EXPECT_TRUE(mpoint_almost_eq(p0p, points_begin_b0[0]));
        EXPECT_TRUE(mpoint_almost_eq(p0d, points_begin_b0[1]));
    }
}

TEST(place_pwlin, segments) {
    // Y-shaped morphology with some discontinuous
    // and zero-length segments.

    segment_tree tree;

    // root branch

    mpoint p0p{0, 0, 0, 1};
    mpoint p0d{1, 0, 0, 1};
    mpoint p1p = p0d;
    mpoint p1d{2, 0, 0, 1};

    msize_t s0 = tree.append(mnpos, p0p, p0d, 0);
    msize_t s1 = tree.append(s0, p1p, p1d, 0);

    // branch A (total length 2)

    mpoint p2p{2, 0, 0, 1};
    mpoint p2d{2, 1, 0, 1};
    mpoint p3p{2, 1, 0, 0.5};
    mpoint p3d{2, 2, 0, 0.5};
    mpoint p4p{8, 9, 7, 1.5}; // some random zero-length segments on the end...
    mpoint p4d = p4p;
    mpoint p5p{3, 1, 3, 0.5};
    mpoint p5d = p5p;

    msize_t s2 = tree.append(s1, p2p, p2d, 0);
    msize_t s3 = tree.append(s2, p3p, p3d, 0);
    msize_t s4 = tree.append(s3, p4p, p4d, 0);
    msize_t s5 = tree.append(s4, p5p, p5d, 0);
    (void)s5;

    // branch B (total length 4)

    mpoint p6p{2, 0, 0, 1};
    mpoint p6d{2, 0, 2, 1};
    mpoint p7p{2, 0, 2, 0.5}; // a zero-length segment in the middle...
    mpoint p7d = p7p;
    mpoint p8p{2, 0, 3, 1};
    mpoint p8d{2, 0, 5, 1};

    msize_t s6 = tree.append(s1, p6p, p6d, 0);
    msize_t s7 = tree.append(s6, p7p, p7d, 0);
    msize_t s8 = tree.append(s7, p8p, p8d, 0);
    (void)s8;

    morphology m(tree);
    mprovider p(m, label_dict{});
    place_pwlin place(m);

    // Thingify a segment expression to work out which branch id is A and
    // which is B.

    mextent s2_extent = thingify(reg::segment(2), p);
    msize_t branch_A = s2_extent.front().branch;

    mextent s6_extent = thingify(reg::segment(6), p);
    msize_t branch_B = s6_extent.front().branch;

    ASSERT_TRUE((branch_A==1 && branch_B==2) || (branch_A==2 && branch_B==1));

    // Region 1: all of branch A, middle section of branch B.

    region r1 = join(reg::branch(branch_A), reg::cable(branch_B, 0.25, 0.75));
    mextent x1 = thingify(r1, p);

    std::vector<msegment> x1min = place.segments(x1);
    std::vector<msegment> x1all = place.all_segments(x1);

    auto seg_id = [](const msegment& s) { return s.id; };

    util::sort_by(x1min, seg_id);
    std::vector<msize_t> x1min_seg_ids = util::assign_from(util::transform_view(x1min, seg_id));

    util::sort_by(x1all, seg_id);
    std::vector<msize_t> x1all_seg_ids = util::assign_from(util::transform_view(x1all, seg_id));

    ASSERT_EQ((std::vector<msize_t>{2, 3, 6, 8}), x1min_seg_ids);
    ASSERT_EQ((std::vector<msize_t>{2, 3, 4, 5, 6, 7, 8}), x1all_seg_ids);

    EXPECT_TRUE(mpoint_almost_eq(p2p, x1all[0].prox));
    EXPECT_TRUE(mpoint_almost_eq(p2d, x1all[0].dist));

    EXPECT_TRUE(mpoint_almost_eq(p3p, x1all[1].prox));
    EXPECT_TRUE(mpoint_almost_eq(p3d, x1all[1].dist));

    EXPECT_TRUE(mpoint_almost_eq(p4p, x1all[2].prox));
    EXPECT_TRUE(mpoint_almost_eq(p4d, x1all[2].dist));

    EXPECT_TRUE(mpoint_almost_eq(p5p, x1all[3].prox));
    EXPECT_TRUE(mpoint_almost_eq(p5d, x1all[3].dist));

    EXPECT_FALSE(mpoint_almost_eq(p6p, x1all[4].prox));
    EXPECT_TRUE(mpoint_almost_eq(p6d, x1all[4].dist));
    EXPECT_TRUE(mpoint_almost_eq(mpoint{2, 0, 1, 1}, x1all[4].prox));

    EXPECT_TRUE(mpoint_almost_eq(p7p, x1all[5].prox));
    EXPECT_TRUE(mpoint_almost_eq(p7d, x1all[5].dist));

    EXPECT_TRUE(mpoint_almost_eq(p8p, x1all[6].prox));
    EXPECT_FALSE(mpoint_almost_eq(p8d, x1all[6].dist));
    EXPECT_TRUE(mpoint_almost_eq(mpoint{2, 0, 4, 1}, x1all[6].dist));

    // Region 2: first half of branch A. Test exclusion of zero-length partial
    // segments.

    region r2 = reg::cable(branch_A, 0., 0.5);
    mextent x2 = thingify(r2, p);

    std::vector<msegment> x2min = place.segments(x2);
    std::vector<msegment> x2all = place.all_segments(x2);

    util::sort_by(x2min, seg_id);
    std::vector<msize_t> x2min_seg_ids = util::assign_from(util::transform_view(x2min, seg_id));

    util::sort_by(x2all, seg_id);
    std::vector<msize_t> x2all_seg_ids = util::assign_from(util::transform_view(x2all, seg_id));

    ASSERT_EQ((std::vector<msize_t>{2}), x2min_seg_ids);
    ASSERT_EQ((std::vector<msize_t>{2, 3}), x2all_seg_ids);

    EXPECT_TRUE(mpoint_almost_eq(p3p, x2all[1].prox));
    EXPECT_TRUE(mpoint_almost_eq(p3p, x2all[1].dist));

    // Region 3: trivial region, midpont of branch B.

    region r3 = reg::cable(branch_B, 0.5, 0.5);
    mextent x3 = thingify(r3, p);

    std::vector<msegment> x3min = place.segments(x3);
    std::vector<msegment> x3all = place.all_segments(x3);

    util::sort_by(x3min, seg_id);
    std::vector<msize_t> x3min_seg_ids = util::assign_from(util::transform_view(x3min, seg_id));

    util::sort_by(x3all, seg_id);
    std::vector<msize_t> x3all_seg_ids = util::assign_from(util::transform_view(x3all, seg_id));

    ASSERT_EQ(1u, x3min_seg_ids.size()); // Could be end of s6, all of s7, or beginning of s8
    ASSERT_EQ((std::vector<msize_t>{6, 7, 8}), x3all_seg_ids);

    EXPECT_TRUE(mpoint_almost_eq(x3min[0].prox, x3min[0].dist));
    EXPECT_TRUE(mpoint_almost_eq(p6d, x3min[0].prox) ||
                mpoint_almost_eq(p7d, x3min[0].prox) ||
                mpoint_almost_eq(p8p, x3min[0].prox));

    EXPECT_TRUE(mpoint_almost_eq(p6d, x3all[0].prox));
    EXPECT_TRUE(mpoint_almost_eq(p6d, x3all[0].dist));
    EXPECT_TRUE(mpoint_almost_eq(p7d, x3all[1].prox));
    EXPECT_TRUE(mpoint_almost_eq(p7d, x3all[1].dist));
    EXPECT_TRUE(mpoint_almost_eq(p8p, x3all[2].prox));
    EXPECT_TRUE(mpoint_almost_eq(p8p, x3all[2].dist));
}

TEST(place_pwlin, nearest) {
    segment_tree tree;

    //  the test morphology:
    //
    //       x=-9        x=9
    //
    //         _        _  y=40
    //          \       /
    //      seg4 \     / seg2
    //  branch 2  \   /     branch 1
    //        seg3 \ /
    //              | y=25
    //              |
    //              |
    //              | branch 0
    //        seg1  |
    //              |
    //              _ y=7
    //
    //              - y=5
    //        seg0  |
    //              _ y=-5

    // Root branch.
    mpoint psoma_p{0, -5, 0, 5};
    mpoint psoma_d{0,  5, 0, 5};

    msize_t ssoma = tree.append(mnpos, psoma_p, psoma_d, 1);

    // Main leg of y, of length 28 μm
    // Note that there is a gap of 2 μm between the end of the soma segment
    mpoint py1_p{0,  7, 0, 1};
    mpoint py1_d{0, 25, 0, 1};

    msize_t sy1 = tree.append(ssoma, py1_p, py1_d, 3);

    // branch 1 of y: translation (9,15) in one segment
    mpoint py2_d{ 9, 40, 0, 1};
    tree.append(sy1, py2_d, 3);

    // branch 2 of y: translation (-9,15) in 2 segments
    mpoint py3_m{-6, 35, 0, 1};
    mpoint py3_d{-9, 40, 0, 1};
    tree.append(tree.append(sy1, py3_m, 3), py3_d, 3);

    morphology m(tree);
    place_pwlin place(m);

    {
        auto [l, d] = place.closest(0, -5, 0);
        EXPECT_EQ((mlocation{0, 0.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(10, -5, 0);
        EXPECT_EQ((mlocation{0, 0.}), l);
        EXPECT_EQ(10., d);
    }
    {
        auto [l, d] = place.closest(0, 0, 0);
        EXPECT_EQ((mlocation{0, 5./28.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(10, 0, 0);
        EXPECT_EQ((mlocation{0, 5./28.}), l);
        EXPECT_EQ(10., d);
    }
    {
        auto [l, d] = place.closest(0, 25, 0);
        EXPECT_EQ((mlocation{0, 1.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(0, 6, 0);
        EXPECT_EQ((mlocation{0, 10./28.}), l);
        EXPECT_EQ(1., d);
    }
    {
        auto [l, d] = place.closest(3, 30, 0);
        EXPECT_EQ((mlocation{1, 1./3.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(-6, 35, 0);
        EXPECT_EQ((mlocation{2, 2./3.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(-9, 40, 0);
        EXPECT_EQ((mlocation{2, 1.}), l);
        EXPECT_EQ(0., d);
    }
    {
        auto [l, d] = place.closest(-9, 41, 0);
        EXPECT_EQ((mlocation{2, 1.}), l);
        EXPECT_EQ(1., d);
    }
}
