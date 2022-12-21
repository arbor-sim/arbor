#include <cmath>
#include <unordered_map>
#include <vector>

#include <arbor/math.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include "util/piecewise.hpp"

#include <gtest/gtest.h>
#include "common.hpp"
#include "../common_cells.hpp"
#include "morph_pred.hpp"

using namespace arb;
using embedding = embed_pwlin;

using testing::mlocation_eq;
using testing::cable_eq;

TEST(embedding, segments_and_branch_length) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<mpoint>;
    using loc = mlocation;

    // A single unbranched cable with 5 sample points.
    // The cable has length 10 μm, with samples located at
    // 0 μm, 1 μm, 3 μm, 7 μm and 10 μm.
    {
        pvec parents = {mnpos, 0, 1, 2, 3};
        svec points = {
            { 0,  0,  0,  2},
            { 1,  0,  0,  2},
            { 3,  0,  0,  2},
            { 7,  0,  0,  2},
            {10,  0,  0,  2},
        };
        morphology m(segments_from_points(points, parents));

        embedding em(m);

        auto nloc = 5u;
        EXPECT_EQ(nloc, em.segment_ends().size());
        const auto& locs = em.segment_ends();
        EXPECT_EQ(nloc, locs.size());
        EXPECT_TRUE(mlocation_eq(locs[0], (loc{0,0})));
        EXPECT_TRUE(mlocation_eq(locs[1], (loc{0,0.1})));
        EXPECT_TRUE(mlocation_eq(locs[2], (loc{0,0.3})));
        EXPECT_TRUE(mlocation_eq(locs[3], (loc{0,0.7})));
        EXPECT_TRUE(mlocation_eq(locs[4], (loc{0,1})));

        EXPECT_EQ(10., em.branch_length(0));
    }

    // Zero-length branch?
    // Four samples - point indices:
    //
    //      0
    //     1 2
    //        3
    //
    // Point 0, 2, and 3 colocated.
    // Expect all but most distal segment on zero length branch
    // to have cable (bid, 0, 0), and the most distal to have (bid, 0, 1).
    {
        pvec parents = {mnpos, 0, 0, 2};

        svec points = {
            {  0,  0,  3,  2},
            { 10,  0,  3,  2},
            {  0,  0,  3,  2},
            {  0,  0,  3,  2},
        };
        morphology m(segments_from_points(points, parents));

        ASSERT_EQ(2u, m.num_branches());

        embedding em(m);
        const auto& locs = em.segment_ends();
        ASSERT_EQ(5u, locs.size());
        EXPECT_TRUE(mlocation_eq(locs[0], (loc{0,0})));
        EXPECT_TRUE(mlocation_eq(locs[1], (loc{0,1})));
        EXPECT_TRUE(mlocation_eq(locs[2], (loc{1,0})));
        EXPECT_TRUE(mlocation_eq(locs[3], (loc{1,0})));
        EXPECT_TRUE(mlocation_eq(locs[4], (loc{1,1})));

        EXPECT_TRUE(cable_eq(mcable{0, 0, 1}, em.segment(0)));
        EXPECT_TRUE(cable_eq(mcable{1, 0, 0}, em.segment(1)));
        EXPECT_TRUE(cable_eq(mcable{1, 0, 1}, em.segment(2)));

        EXPECT_EQ(10, em.branch_length(0));
        EXPECT_EQ(0, em.branch_length(1));
    }

    // Eight samples - point indices:
    //
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    {
        pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};

        svec points = {
            {  0,  0,  0,  2},
            { 10,  0,  0,  2},
            {100,  0,  0,  2},
            {  0, 10,  0,  2},
            {  0,100,  0,  2},
            {100,100,  0,  2},
            {  0,130,  0,  2},
            {  0,300,  0,  2},
        };
        morphology m(segments_from_points(points, parents));

        ASSERT_EQ(4u, m.num_branches());

        embedding em(m);

        const auto& locs = em.segment_ends();
        EXPECT_TRUE(mlocation_eq(locs[0], (loc{0,0})));
        EXPECT_TRUE(mlocation_eq(locs[1], (loc{0,0.1})));
        EXPECT_TRUE(mlocation_eq(locs[2], (loc{0,1})));
        EXPECT_TRUE(mlocation_eq(locs[3], (loc{1,0})));
        EXPECT_TRUE(mlocation_eq(locs[4], (loc{1,0.1})));
        EXPECT_TRUE(mlocation_eq(locs[5], (loc{1,1})));
        EXPECT_TRUE(mlocation_eq(locs[6], (loc{2,0})));
        EXPECT_TRUE(mlocation_eq(locs[7], (loc{2,1})));
        EXPECT_TRUE(mlocation_eq(locs[8], (loc{3,0})));
        EXPECT_TRUE(mlocation_eq(locs[9], (loc{3,0.15})));
        EXPECT_TRUE(mlocation_eq(locs[10], (loc{3,1})));

        EXPECT_TRUE(cable_eq(mcable{0, 0.  , 0.1 }, em.segment(0)));
        EXPECT_TRUE(cable_eq(mcable{0, 0.1 , 1.  }, em.segment(1)));
        EXPECT_TRUE(cable_eq(mcable{1, 0.  , 0.1 }, em.segment(2)));
        EXPECT_TRUE(cable_eq(mcable{1, 0.1 , 1.  }, em.segment(3)));
        EXPECT_TRUE(cable_eq(mcable{2, 0.  , 1.  }, em.segment(4)));
        EXPECT_TRUE(cable_eq(mcable{3, 0.  , 0.15}, em.segment(5)));
        EXPECT_TRUE(cable_eq(mcable{3, 0.15, 1.  }, em.segment(6)));

        EXPECT_EQ(100., em.branch_length(0));
        EXPECT_EQ(100., em.branch_length(1));
        EXPECT_EQ(100., em.branch_length(2));
        EXPECT_EQ(200., em.branch_length(3));
    }
}

// TODO: integrator tests

TEST(embedding, partial_branch_length) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<mpoint>;
    using util::pw_elements;

    pvec parents = {mnpos, 0, 1, 2, 2};
    svec points = {
        { 0,  0,  0, 10},
        {10,  0,  0, 20},
        {30,  0,  0, 10},
        {30, 10,  0,  5},
        {30,  0, 50,  5}
    };

    morphology m(segments_from_points(points, parents));
    embedding em(m);

    EXPECT_DOUBLE_EQ(30., em.branch_length(0));
    EXPECT_DOUBLE_EQ(30., em.integrate_length(mcable{0, 0., 1.}));
    EXPECT_DOUBLE_EQ(15., em.integrate_length(mcable{0, 0.25, 0.75}));

    EXPECT_DOUBLE_EQ(10., em.branch_length(1));
    EXPECT_DOUBLE_EQ(10., em.integrate_length(mcable{1, 0., 1.}));
    EXPECT_DOUBLE_EQ(7.5, em.integrate_length(mcable{1, 0.25, 1.0}));

    // Expect 2*0.25+3*0.5 = 2.0 times corresponding cable length.
    pw_elements<double> pw({0.25, 0.5, 1.}, {2., 3.});
    EXPECT_DOUBLE_EQ(20., em.integrate_length(1, pw));

    // Distamce between points on different branches:
    ASSERT_EQ(3u, m.num_branches());
    ASSERT_EQ(0u, m.branch_parent(2));
    EXPECT_DOUBLE_EQ(em.integrate_length(mcable{0, 0.75, 1.})+em.integrate_length(mcable{2, 0, 0.5}),
        em.integrate_length(mlocation{0, 0.75}, mlocation{2, 0.5}));
}

TEST(embedding, partial_area) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<mpoint>;
    using util::pw_elements;
    using testing::near_relative;

    pvec parents = {mnpos, 0, 1, 2, 2};
    svec points = {
        { 0,  0,  0, 10},
        {10,  0,  0, 20},
        {30,  0,  0, 10},
        {30, 10,  0,  5},
        {30,  0, 50,  5}
    };

    morphology m(segments_from_points(points, parents));
    embedding em(m);

    // Cable 1: single truncated cone, length L = 10,
    // radius r₀ = 10 (pos 0) to r₁ = 5 (pos 1).
    //
    // Expect cable area = 2πLr√(1 + m²)
    // where m = δr/L and r = (r₀+r₁)/2 = r₀ + δr/2.

    constexpr double pi = math::pi<double>;
    double cable1_area = 2*pi*10*7.5*std::sqrt(1.25);

    constexpr double reltol = 1e-10;

    EXPECT_TRUE(near_relative(cable1_area, em.integrate_area(mcable{1, 0., 1.}), reltol));

    // Weighted area within cable 0:
    // a)  proportional segment [0.1, 0.3]:
    //         truncated cone length 6,
    //         r₀ = 13; r₁ = 19, slope = 1
    //
    // b)  proportional segment [0.3, 0.9]:
    //         truncated cone length 1,
    //         r₀ = 19, r₁ = 20, slope = 1
    //         truncated cone length 17
    //         r₀ = 20, r₁ = 11.5, slope = -0.5

    EXPECT_TRUE(near_relative(13., em.radius(mlocation{0, 0.1}), reltol));
    EXPECT_TRUE(near_relative(19., em.radius(mlocation{0, 0.3}), reltol));
    EXPECT_TRUE(near_relative(11.5, em.radius(mlocation{0, 0.9}), reltol));

    pw_elements<double> pw({0.1, 0.3, 0.9}, {5., 7.});
    double sub_area1 = pi*6*(13+19)*std::sqrt(2);
    double sub_area2 = pi*1*(19+20)*std::sqrt(2);
    double sub_area3 = pi*17*(20+11.5)*std::sqrt(1.25);

    EXPECT_TRUE(near_relative(sub_area1, em.integrate_area(mcable{0, 0.1, 0.3}), reltol));
    EXPECT_TRUE(near_relative(sub_area2, em.integrate_area(mcable{0, 0.3, 1/3.}), reltol));
    EXPECT_TRUE(near_relative(sub_area3, em.integrate_area(mcable{0, 1/3., 0.9}), reltol));

    double expected_pw_area = 5.*sub_area1+7.*(sub_area2+sub_area3);
    EXPECT_TRUE(near_relative(expected_pw_area, em.integrate_area(0, pw), reltol));

    // Area between points on different branches:
    ASSERT_EQ(3u, m.num_branches());
    ASSERT_EQ(0u, m.branch_parent(2));
    EXPECT_TRUE(near_relative(
        em.integrate_area(mcable{0, 0.8, 1.})+em.integrate_area(mcable{2, 0, 0.3}),
        em.integrate_area(mlocation{0, 0.8}, mlocation{2, 0.3}), reltol));

    // Integrated inverse cross-sectional area in cable 1 from 0.1 to 0.4:
    // radius r₀ = 9.5, r₁ = 8, length = 3.

    double expected_ixa = 3/(9.5*8)/pi;
    EXPECT_TRUE(near_relative(expected_ixa, em.integrate_ixa(mcable{1, 0.1, 0.4}), reltol));
}

TEST(embedding, area_0_length_segment) {
    using testing::near_relative;
    constexpr double pi = math::pi<double>;
    constexpr double reltol = 1e-10;

    segment_tree t1, t2;

    t1.append(mnpos, { 0, 0, 0, 10}, {10, 0, 0, 10}, 0);
    t1.append(0,     {10, 0, 0, 20}, {30, 0, 0, 20}, 0);

    t2.append(mnpos, { 0, 0, 0, 10}, {10, 0, 0, 10}, 0);
    t2.append(0,     {10, 0, 0, 10}, {10, 0, 0, 20}, 0);
    t2.append(1,     {10, 0, 0, 20}, {30, 0, 0, 20}, 0);

    embedding em1{morphology(t1)}, em2{morphology(t2)};

    double a1 = em1.integrate_area(mcable{0, 0, 1});
    double expected_a1 = 2*pi*(10*10+20*20);
    EXPECT_TRUE(near_relative(a1, expected_a1, reltol));

    // The second morphology includes the anulus joining the
    // first and last segment.

    double a2 = em2.integrate_area(mcable{0, 0, 1});
    double expected_a2 = expected_a1 + pi*(20*20-10*10);
    EXPECT_TRUE(near_relative(a2, expected_a2, reltol));
}

TEST(embedding, small_radius) {
    using testing::near_relative;
    constexpr double pi = math::pi<double>;
    constexpr double reltol = 1e-10;

    segment_tree t;
    t.append(mnpos, { 0, 0, 0, 10}, {10, 0, 0, 0.00001}, 0);
    t.append(0,     {10, 0, 0, 40}, {20, 0, 0, 40}, 0);

    embedding em{morphology(t)};

    // Integrated inverse cross-sectional area in segment 1
    // corresponding to cable(0, 0.5, 1):
    // radius r₀ = 40, r₁ = 40, length = 10.

    double expected_ixa = 10/(40*40*pi);
    double computed_ixa = em.integrate_ixa(mcable{0, 0.5, 1.0});
    ASSERT_FALSE(std::isnan(computed_ixa));
    EXPECT_TRUE(near_relative(expected_ixa, computed_ixa, reltol));
}

TEST(embedding, zero_radius) {
    using testing::near_relative;
    constexpr double pi = math::pi<double>;
    constexpr double reltol = 1e-10;

    segment_tree t;
    t.append(mnpos, { 0, 0, 0, 10}, {10, 0, 0, 0}, 0);
    t.append(0,     {10, 0, 0, 40}, {20, 0, 0, 40}, 0);

    embedding em{morphology(t)};

    // Integrated inverse cross-sectional area in segment 1
    // corresponding to cable(0, 0.5, 1):
    // radius r₀ = 40, r₁ = 40, length = 10.

    double expected_ixa = 10/(40*40*pi);
    double computed_ixa = em.integrate_ixa(mcable{0, 0.5, 1.0});
    ASSERT_FALSE(std::isnan(computed_ixa));
    EXPECT_TRUE(near_relative(expected_ixa, computed_ixa, reltol));

    // Integrating over the zero radius point though should give us
    // INFINITY.

    double infinite_ixa = em.integrate_ixa(mcable{0, 0.25, 0.75});
    ASSERT_FALSE(std::isnan(infinite_ixa));
    EXPECT_TRUE(std::isinf(infinite_ixa));

    // Integrating to the zero radius point should also give us
    // INFINITY, because we integrate over closed intervals.

    double also_infinite_ixa = em.integrate_ixa(mcable{0, 0.25, 0.5});
    ASSERT_FALSE(std::isnan(also_infinite_ixa));
    EXPECT_TRUE(std::isinf(also_infinite_ixa));

    // Should be able to integrate ixa over a tree that starts
    // with a zero radius.

    segment_tree t2;
    t2.append(mnpos, { 0, 0, 0, 0}, {10, 0, 0, 10}, 0);
    embedding em2{morphology(t2)};

    expected_ixa = 5/(10*5*pi);
    computed_ixa = em2.integrate_ixa(mcable{0, 0.5, 1.0});
    ASSERT_FALSE(std::isnan(computed_ixa));
    EXPECT_TRUE(near_relative(expected_ixa, computed_ixa, reltol));

    EXPECT_TRUE(std::isinf(em2.integrate_ixa(mcable{0, 0, 0.5})));
}
