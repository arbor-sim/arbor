#include <cmath>
#include <vector>

#include <arbor/math.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

#include "util/piecewise.hpp"

#include "../test/gtest.h"
#include "common.hpp"

using namespace arb;
using embedding = embed_pwlin;

::testing::AssertionResult location_eq(const morphology& m, mlocation a, mlocation b) {
    a = canonical(m, a);
    b = canonical(m, b);

    if (a.branch!=b.branch) {
        return ::testing::AssertionFailure()
            << "branch ids " << a.branch << " and " << b.branch << " differ";
    }

    using FP = testing::internal::FloatingPoint<double>;
    if (FP(a.pos).AlmostEquals(FP(b.pos))) {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure()
            << "location positions " << a.pos << " and " << b.pos << " differ";
    }
}

TEST(embedding, samples_and_branch_length) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using loc = mlocation;

    // A single unbranched cable with 5 sample points.
    // The cable has length 10 μm, with samples located at
    // 0 μm, 1 μm, 3 μm, 7 μm and 10 μm.
    {
        pvec parents = {mnpos, 0, 1, 2, 3};
        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{  1,  0,  0,  2}, 1},
            {{  3,  0,  0,  2}, 1},
            {{  7,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 1},
        };
        sample_tree sm(samples, parents);
        morphology m(sm, false);

        embedding em(m);

        EXPECT_TRUE(location_eq(m, em.sample_location(0), (loc{0,0})));
        EXPECT_TRUE(location_eq(m, em.sample_location(1), (loc{0,0.1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(2), (loc{0,0.3})));
        EXPECT_TRUE(location_eq(m, em.sample_location(3), (loc{0,0.7})));
        EXPECT_TRUE(location_eq(m, em.sample_location(4), (loc{0,1})));

        EXPECT_EQ(10., em.branch_length(0));
    }

    // Eight samples
    //
    //  sample ids:
    //            0
    //           1 3
    //          2   4
    //             5 6
    //                7
    {   // Spherical root.
        pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};

        svec samples = {
            {{  0,  0,  0, 10}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 3},
            {{  0,100,  0,  2}, 3},
            {{100,100,  0,  2}, 3},
            {{  0,200,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        sample_tree sm(samples, parents);
        morphology m(sm, true);
        ASSERT_EQ(5u, m.num_branches());

        embedding em(m);

        EXPECT_TRUE(location_eq(m, em.sample_location(0), (loc{0,0.5})));
        EXPECT_TRUE(location_eq(m, em.sample_location(1), (loc{1,0})));
        EXPECT_TRUE(location_eq(m, em.sample_location(2), (loc{1,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(3), (loc{2,0})));
        EXPECT_TRUE(location_eq(m, em.sample_location(4), (loc{2,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(5), (loc{3,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(6), (loc{4,0.5})));
        EXPECT_TRUE(location_eq(m, em.sample_location(7), (loc{4,1})));

        EXPECT_EQ(20.,  em.branch_length(0));
        EXPECT_EQ(90.,  em.branch_length(1));
        EXPECT_EQ(90.,  em.branch_length(2));
        EXPECT_EQ(100., em.branch_length(3));
        EXPECT_EQ(200., em.branch_length(4));
    }
    {   // No Spherical root
        pvec parents = {mnpos, 0, 1, 0, 3, 4, 4, 6};

        svec samples = {
            {{  0,  0,  0,  2}, 1},
            {{ 10,  0,  0,  2}, 3},
            {{100,  0,  0,  2}, 3},
            {{  0, 10,  0,  2}, 3},
            {{  0,100,  0,  2}, 3},
            {{100,100,  0,  2}, 3},
            {{  0,130,  0,  2}, 3},
            {{  0,300,  0,  2}, 3},
        };
        sample_tree sm(samples, parents);
        morphology m(sm, false);
        ASSERT_EQ(4u, m.num_branches());

        embedding em(m);

        EXPECT_TRUE(location_eq(m, em.sample_location(0), (loc{0,0})));
        EXPECT_TRUE(location_eq(m, em.sample_location(1), (loc{0,0.1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(2), (loc{0,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(3), (loc{1,0.1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(4), (loc{1,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(5), (loc{2,1})));
        EXPECT_TRUE(location_eq(m, em.sample_location(6), (loc{3,0.15})));
        EXPECT_TRUE(location_eq(m, em.sample_location(7), (loc{3,1})));

        EXPECT_EQ(100., em.branch_length(0));
        EXPECT_EQ(100., em.branch_length(1));
        EXPECT_EQ(100., em.branch_length(2));
        EXPECT_EQ(200., em.branch_length(3));
    }
}

// TODO: integrator tests


TEST(embedding, partial_branch_length) {
    using pvec = std::vector<msize_t>;
    using svec = std::vector<msample>;
    using util::pw_elements;

    pvec parents = {mnpos, 0, 1, 2, 2};
    svec samples = {
        {{  0,  0,  0, 10}, 1},
        {{ 10,  0,  0, 20}, 1},
        {{ 30,  0,  0, 10}, 1},
        {{ 30, 10,  0, 5},  2},
        {{ 30,  0, 50, 5},  2}
    };

    morphology m(sample_tree(samples, parents), false);
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
    using svec = std::vector<msample>;
    using util::pw_elements;
    using testing::near_relative;

    pvec parents = {mnpos, 0, 1, 2, 2};
    svec samples = {
        {{  0,  0,  0, 10}, 1},
        {{ 10,  0,  0, 20}, 1},
        {{ 30,  0,  0, 10}, 1},
        {{ 30, 10,  0, 5},  2},
        {{ 30,  0, 50, 5},  2}
    };

    morphology m(sample_tree(samples, parents), false);
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
