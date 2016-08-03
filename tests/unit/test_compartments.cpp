#include <cmath>

#include "gtest.h"

#include "../src/compartment.hpp"
#include "../src/util.hpp"

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

TEST(compartments, compartment_iterator)
{
    // iterator with arguments
    //  idx = 0
    //  len = 2.5
    //  rad = 1
    //  delta_rad = 2
    nest::mc::compartment_iterator it(0, 2.5, 1, 2);

    // so each time the iterator is incremented
    //   idx is incremented by 1
    //   len is unchanged
    //   rad is incremented by 2

    // check the prefix increment
    ++it;
    {
        auto c = *it;
        EXPECT_EQ(c.index, 1u);
        EXPECT_EQ(left(c.radius), 3.0);
        EXPECT_EQ(right(c.radius), 5.0);
        EXPECT_EQ(c.length, 2.5);
    }

    // check postfix increment

    // returned iterator should be unchanged
    {
        auto c = *(it++);
        EXPECT_EQ(c.index, 1u);
        EXPECT_EQ(left(c.radius), 3.0);
        EXPECT_EQ(right(c.radius), 5.0);
        EXPECT_EQ(c.length, 2.5);
    }
    // while the iterator itself was updated
    {
        auto c = *it;
        EXPECT_EQ(c.index, 2u);
        EXPECT_EQ(left(c.radius), 5.0);
        EXPECT_EQ(right(c.radius), 7.0);
        EXPECT_EQ(c.length, 2.5);
    }

    // check that it can be copied and compared
    {
        // copy iterator
        auto it2 = it;
        auto c = *it2;
        EXPECT_EQ(c.index, 2u);
        EXPECT_EQ(left(c.radius), 5.0);
        EXPECT_EQ(right(c.radius), 7.0);
        EXPECT_EQ(c.length, 2.5);

        // comparison
        EXPECT_EQ(it2, it);
        it2++;
        EXPECT_NE(it2, it);

        // check the copy has updated correctly when incremented
        c= *it2;
        EXPECT_EQ(c.index, 3u);
        EXPECT_EQ(left(c.radius), 7.0);
        EXPECT_EQ(right(c.radius), 9.0);
        EXPECT_EQ(c.length, 2.5);
    }
}

TEST(compartments, compartment_range)
{
    {
        nest::mc::compartment_range rng(10, 1.0, 2.0, 10.);

        EXPECT_EQ((*rng.begin()).index, 0u);
        EXPECT_EQ((*rng.end()).index, 10u);
        EXPECT_NE(rng.begin(), rng.end());

        unsigned count = 0;
        for(auto c : rng) {
            EXPECT_EQ(c.index, count);
            auto er = 1.0 + double(count)/10.;
            EXPECT_TRUE(std::fabs(left(c.radius)-er)<1e-15);
            EXPECT_TRUE(std::fabs(right(c.radius)-(er+0.1))<1e-15);
            EXPECT_EQ(c.length, 1.0);
            ++count;
        }
        EXPECT_EQ(count, 10u);
    }

    // test case of zero length range
    {
        nest::mc::compartment_range rng(0, 1.0, 1.0, 0.);

        EXPECT_EQ(rng.begin(), rng.end());
    }
}
