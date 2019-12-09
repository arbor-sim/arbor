#include <iostream>
#include <vector>

#include <arbor/morph/primitives.hpp>

#include "../gtest.h"

using namespace arb;

::testing::AssertionResult mpoint_eq(const mpoint& a, const mpoint& b) {
    if (a.x==b.x && a.y==b.y && a.z==b.z && a.radius==b.radius) {
        return ::testing::AssertionSuccess();
    }
    else {
        return ::testing::AssertionFailure()
            << "mpoints " << a << " and " << b << " differ.";
    }
}

TEST(morph_primitives, lerp) {
    mpoint a{1., 2., 3., 4.};
    mpoint b{2., 4., 7., 12.};

    EXPECT_TRUE(mpoint_eq(a, lerp(a, b, 0)));
    EXPECT_TRUE(mpoint_eq(b, lerp(a, b, 1)));

    EXPECT_TRUE(mpoint_eq(mpoint{1.5, 3., 5., 8.}, lerp(a, b, 0.5)));
    EXPECT_TRUE(mpoint_eq(mpoint{1.25, 2.5, 4., 6.}, lerp(a, b, 0.25)));
}

TEST(morph_primitives, is_collocated) {
    EXPECT_TRUE(is_collocated(mpoint{1., 2., 3., 4.}, mpoint{1., 2., 3., 4.}));
    EXPECT_TRUE(is_collocated(mpoint{1., 2., 3., 4.}, mpoint{1., 2., 3., 4.5}));
    EXPECT_FALSE(is_collocated(mpoint{1., 2., 2.5, 4.}, mpoint{1., 2., 3., 4}));
    EXPECT_FALSE(is_collocated(mpoint{1., 2.5, 3., 4.}, mpoint{1., 2., 3., 4}));
    EXPECT_FALSE(is_collocated(mpoint{2.5, 2, 3., 4.}, mpoint{1., 2., 3., 4}));

    EXPECT_TRUE(is_collocated(msample{{1., 2., 3., 4.},3}, msample{{1., 2., 3., 4.},3}));
    EXPECT_TRUE(is_collocated(msample{{1., 2., 3., 4.},3}, msample{{1., 2., 3., 4.5},1}));
    EXPECT_FALSE(is_collocated(msample{{1., 2., 2.5, 4.},3}, msample{{1., 2., 3., 4},3}));
    EXPECT_FALSE(is_collocated(msample{{1., 2.5, 3., 4.},3}, msample{{1., 2., 3., 4},3}));
    EXPECT_FALSE(is_collocated(msample{{2.5, 2, 3., 4.},3}, msample{{1., 2., 3., 4},3}));
}

TEST(morph_primitives, distance) {
    // 2² + 3² + 6² = 7²
    EXPECT_DOUBLE_EQ(7., distance(mpoint{1.5, 2.5, 3.5, 4.5}, mpoint{3.5, 5.5, 9.5, 0.31}));
}

TEST(morph_primitives, join_intersect_sum) {
    auto ml = [](const std::vector<int>& bids) {
        mlocation_list L;
        for (msize_t b: bids) L.push_back({b, 0});
        return L;
    };

    using ll = mlocation_list;

    {
        ll lhs{};
        ll rhs{};
        EXPECT_EQ(sum(lhs, rhs), ll{});
        EXPECT_EQ(join(lhs, rhs), ll{});
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs{};
        ll rhs = ml({0,1});
        EXPECT_EQ(sum(lhs, rhs), rhs);
        EXPECT_EQ(join(lhs, rhs), rhs);
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs = ml({1});
        ll rhs = ml({1});
        EXPECT_EQ(sum(lhs,  rhs), ml({1,1}));
        EXPECT_EQ(join(lhs, rhs), ml({1}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({1,1});
        ll rhs = ml({1});
        EXPECT_EQ(sum(lhs,  rhs), ml({1,1,1}));
        EXPECT_EQ(join(lhs, rhs), ml({1,1}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs,  rhs), ml({0,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ll{});
    }
    {
        ll lhs = ml({0,1,3});
        ll rhs = ml({0,1,3});
        EXPECT_EQ(sum(lhs, rhs), ml({0,0,1,1,3,3}));
        EXPECT_EQ(join(lhs, rhs), lhs);
        EXPECT_EQ(intersection(lhs, rhs), lhs);
    }
    {
        ll lhs = ml({0,1,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs, rhs), ml({0,1,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,1,1,3});
        ll rhs = ml({1,2});
        EXPECT_EQ(sum(lhs, rhs), ml({0,1,1,1,2,3}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,1,2,3}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1}));
    }
    {
        ll lhs = ml({0,1,1,3,5,5,12});
        ll rhs = ml({1,2,2,5,5,5});
        EXPECT_EQ(sum(lhs, rhs),  ml({0,1,1,1,2,2,3,5,5,5,5,5,12}));
        EXPECT_EQ(join(lhs, rhs), ml({0,1,1,2,2,3,5,5,5,12}));
        EXPECT_EQ(intersection(lhs, rhs), ml({1,5,5}));
    }
}
