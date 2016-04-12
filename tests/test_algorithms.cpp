#include "gtest.h"

#include <vector>

#include "../src/algorithms.hpp"
#include "../src/util.hpp"

TEST(algorithms, sum)
{
    // sum of 10 times 2 is 20
    std::vector<int> v1(10, 2);
    EXPECT_EQ(10*2, nest::mc::algorithms::sum(v1));

    // make an array 1:20 and sum it up using formula for arithmetic sequence
    std::vector<int> v2(20);
    std::iota(v2.begin(), v2.end(), 1);
    auto n = 20;
    EXPECT_EQ((n+1)*n/2, nest::mc::algorithms::sum(v2));
}

TEST(algorithms, make_index)
{
    {
        std::vector<int> v(10, 1);
        auto index = nest::mc::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), nest::mc::algorithms::sum(v));
    }

    {
        std::vector<int> v(10);
        std::iota(v.begin(), v.end(), 1);
        auto index = nest::mc::algorithms::make_index(v);

        EXPECT_EQ(index.size(), 11u);
        EXPECT_EQ(index.front(), 0);
        EXPECT_EQ(index.back(), nest::mc::algorithms::sum(v));
    }
}

TEST(algorithms, minimal_degree)
{
    {
        std::vector<int> v = {0};
        EXPECT_TRUE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 1, 2, 3, 4};
        EXPECT_TRUE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 3, 4};
        EXPECT_TRUE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 0, 4};
        EXPECT_TRUE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 0, 1, 2, 0, 4, 5, 4};
        EXPECT_TRUE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {1};
        EXPECT_FALSE(nest::mc::algorithms::is_minimal_degree(v));
    }

    {
        std::vector<int> v = {0, 2};
        EXPECT_FALSE(nest::mc::algorithms::is_minimal_degree(v));
    }
}

TEST(algorithms, is_strictly_monotonic_increasing)
{
    EXPECT_TRUE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0, 1, 2, 3}
        )
    );
    EXPECT_TRUE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 42, 89}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{0, 0}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 20, 89}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_increasing(
            std::vector<int>{3, 2, 1, 0}
        )
    );
}

TEST(algorithms, is_strictly_monotonic_decreasing)
{
    EXPECT_TRUE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{3, 2, 1, 0}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0, 1, 2, 3}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{8, 20, 42, 89}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{0, 0}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_strictly_monotonic_decreasing(
            std::vector<int>{8, 20, 20, 89}
        )
    );
}

TEST(algorithms, is_positive)
{
    EXPECT_TRUE(
        nest::mc::algorithms::is_positive(
            std::vector<int>{}
        )
    );
    EXPECT_TRUE(
        nest::mc::algorithms::is_positive(
            std::vector<int>{3, 2, 1}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_positive(
            std::vector<int>{3, 2, 1, 0}
        )
    );
    EXPECT_FALSE(
        nest::mc::algorithms::is_positive(
            std::vector<int>{-1}
        )
    );
}
