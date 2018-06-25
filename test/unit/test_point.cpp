#include <cmath>
#include <limits>

#include "../gtest.h"

#include <arbor/point.hpp>

using namespace arb;

TEST(point, construction)
{
    {
        // default constructor
        point<float> p;

        EXPECT_FALSE(p.is_set());

        // expect NaN, which returns false when comparing for equality
        EXPECT_NE(p.x, p.x);
        EXPECT_NE(p.y, p.y);
        EXPECT_NE(p.z, p.z);
    }

    {
        // initializer list
        point<float> p{1, 2, 3};
        EXPECT_EQ(p.x, 1.);
        EXPECT_EQ(p.y, 2.);
        EXPECT_EQ(p.z, 3.);
    }

    {
        // explicit call to constructor
        point<double> p1(1, 2, 3);

        EXPECT_EQ(p1.x, 1.);
        EXPECT_EQ(p1.y, 2.);
        EXPECT_EQ(p1.z, 3.);

        // copy constructor
        auto p2 = p1;
        EXPECT_EQ(p1, p2);
    }
}

TEST(point, constexpr)
{
    // perform test using constexpr
    constexpr point<double> p1(1, 2, 3);
    constexpr point<double> p2(1, 2, 3);

    constexpr auto p = p1 + p2;

    static_assert(p.x == p1.x+p2.x, "failed x-component");
    static_assert(p.y == p1.y+p2.y, "failed y-component");
    static_assert(p.z == p1.z+p2.z, "failed z-component");
}

TEST(point, addition)
{
    // perform test using constexpr
    point<double> p1(1, 2, 3);
    point<double> p2(1, 2, 3);

    auto p = p1 + p2;

    EXPECT_EQ(point<double>(2, 4, 6), p);
}

TEST(point, subtraction)
{
    // perform test using constexpr
    point<double> p1(1, 2, 3);
    point<double> p2(1, 2, 3);

    auto p = p1 - p2;

    EXPECT_EQ(point<double>(0, 0, 0), p);
}

TEST(point, scalar_prod)
{
    // perform test using constexpr
    point<double> p1(1, 2, 3);

    auto p = 0.5 * p1;

    EXPECT_EQ(point<double>(0.5, 1.0, 1.5), p);
}

TEST(point, norm)
{
    // don't use constexpr, because the sqrt is not constexpr
    point<double> p1(1, 1, 1);
    point<double> p2(1, 2, 3);

    EXPECT_EQ(norm(p1), std::sqrt(3.));
    EXPECT_EQ(norm(p2), std::sqrt(1.+4.+9.));
}

TEST(point, dot)
{
    // perform test using constexpr
    constexpr point<double> p1(1, -1, 1);
    constexpr point<double> p2(1, 2, 3);

    static_assert(dot(p1,p2)==2., "unable to perform constexpr dot product");

    EXPECT_EQ(dot(p1,p2), 2.);
}
