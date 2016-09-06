#include <cmath>

#include "gtest.h"

#include <compartment.hpp>
#include <util.hpp>

using nest::mc::util::left;
using nest::mc::util::right;

// not much to test here: just test that values passed into the constructor
// are correctly stored in members
TEST(compartments, compartment)
{
    {
        nest::mc::compartment c(100, 1.2, 2.1, 2.2);
        EXPECT_EQ(c.index, 100u);
        EXPECT_EQ(c.length, 1.2);
        EXPECT_EQ(left(c.radius), 2.1);
        EXPECT_EQ(right(c.radius), 2.2);

        auto c2 = c;
        EXPECT_EQ(c2.index, 100u);
        EXPECT_EQ(c2.length, 1.2);
        EXPECT_EQ(left(c2.radius), 2.1);
        EXPECT_EQ(right(c2.radius), 2.2);
    }

    {
        nest::mc::compartment c{100, 1, 2, 3};
        EXPECT_EQ(c.index, 100u);
        EXPECT_EQ(c.length, 1.);
        EXPECT_EQ(left(c.radius), 2.);
        EXPECT_EQ(right(c.radius), 3.);
    }
}

TEST(compartments, make_compartment_range)
{
    using namespace nest::mc;
    auto rng = make_compartment_range(10, 1.0, 2.0, 10.);

    EXPECT_EQ((*rng.begin()).index, 0u);
    EXPECT_EQ((*rng.end()).index, 10u);
    EXPECT_NE(rng.begin(), rng.end());

    unsigned count = 0;
    for (auto c : rng) {
        EXPECT_EQ(c.index, count);
        auto er = 1.0 + double(count)/10.;
        EXPECT_DOUBLE_EQ(left(c.radius), er);
        EXPECT_DOUBLE_EQ(right(c.radius), er+0.1);
        EXPECT_EQ(c.length, 1.0);
        ++count;
    }
    EXPECT_EQ(count, 10u);

    // test case of zero length range
    auto rng_empty = make_compartment_range(0, 1.0, 1.0, 0.);
    EXPECT_EQ(rng_empty.begin(), rng_empty.end());
}
