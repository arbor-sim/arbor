#include <vector>

#include <gtest/gtest.h>

#include <backends/gpu/forest.hpp>

using namespace arb::gpu;
TEST(forest, is_strictly_monotonic_increasing)
{
    EXPECT_TRUE(
        is_strictly_monotonic_increasing(
            std::vector<int>{0}
        )
    );
    EXPECT_TRUE(
        is_strictly_monotonic_increasing(
            std::vector<int>{0, 1, 2, 3}
        )
    );
    EXPECT_TRUE(
        is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 42, 89}
        )
    );
    EXPECT_FALSE(
        is_strictly_monotonic_increasing(
            std::vector<int>{0, 0}
        )
    );
    EXPECT_FALSE(
        is_strictly_monotonic_increasing(
            std::vector<int>{8, 20, 20, 89}
        )
    );
    EXPECT_FALSE(
        is_strictly_monotonic_increasing(
            std::vector<int>{3, 2, 1, 0}
        )
    );
}

TEST(forest, has_contiguous_compartments)
{
    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 7 4
    //    /     \.
    //   5       6
    //
    EXPECT_FALSE(
        has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 2, 3, 4, 2}
        )
    );

    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 6 5
    //    /     \.
    //   4       7
    //
    EXPECT_FALSE(
        has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 2, 5}
        )
    );

    //
    //       0
    //       |
    //       1
    //       |
    //       2
    //      /|\.
    //     3 7 5
    //    /     \.
    //   4       6
    //
    EXPECT_TRUE(
        has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 5, 2}
        )
    );

    //
    //         0
    //         |
    //         1
    //        / \.
    //       2   7
    //      / \.
    //     3   5
    //    /     \.
    //   4       6
    //
    EXPECT_TRUE(
        has_contiguous_compartments(
            std::vector<int>{0, 0, 1, 2, 3, 2, 5, 1}
        )
    );

    //
    //     0
    //    / \.
    //   1   2
    //  / \.
    // 3   4
    //
    EXPECT_TRUE(
        has_contiguous_compartments(
            std::vector<int>{0, 0, 0, 1, 1}
        )
    );

    // Soma-only list
    EXPECT_TRUE(
        has_contiguous_compartments(
            std::vector<int>{0}
        )
    );

    // Empty list
    EXPECT_TRUE(
        has_contiguous_compartments(
            std::vector<int>{}
        )
    );
}

TEST(forest, branches) {
    using namespace arb;
    {
        //
        //        0                        0
        //       /|\.                     /|\.
        //      1 4 6                    1 2 3
        //     /  |  \.           =>        / \.
        //    2   5   7                    4   5
        //   /         \.
        //  3           8
        //             / \.
        //            9   11
        //           /     \.
        //          10      12
        //                   \.
        //                    13
        //
        std::vector<int> parent_index =
            { 0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12 };
        std::vector<int> expected_branches =
            { 0, 1, 4, 6, 9, 11, 14 };
        std::vector<int> expected_parent_index =
            { 0, 0, 0, 0, 3, 3 };

        auto actual_branches = branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);

        // Check expand_branches
        auto expanded = gpu::expand_branches(actual_branches);
        EXPECT_EQ(parent_index.size(), expanded.size());
        for (std::size_t i = 0; i < parent_index.size(); ++i) {
            auto it =  std::find_if(
                actual_branches.begin(), actual_branches.end(),
                [i](const int& v) { return v > (int)i; }
            );
            auto branch = it - actual_branches.begin() - 1;
            EXPECT_EQ(branch,expanded[i]);
        }
    }
    {
        //
        //    0      0
        //    |      |
        //    1  =>  1
        //    |
        //    2
        //    |
        //    3
        //
        std::vector<int> parent_index          = { 0, 0, 1, 2 };
        std::vector<int> expected_branches     = { 0, 1, 4 };
        std::vector<int> expected_parent_index = { 0, 0 };

        auto actual_branches = gpu::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            gpu::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }

    {
        //
        //    0           0
        //    |           |
        //    1     =>    1
        //    |          / \.
        //    2         2   3
        //   / \.
        //  3   4
        //       \.
        //        5
        //
        std::vector<int> parent_index          = { 0, 0, 1, 2, 2, 4 };
        std::vector<int> expected_branches     = { 0, 1, 3, 4, 6 };
        std::vector<int> expected_parent_index = { 0, 0, 1, 1 };

        auto actual_branches = gpu::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            gpu::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }

    {
        std::vector<int> parent_index          = { 0 };
        std::vector<int> expected_branches     = { 0, 1 };
        std::vector<int> expected_parent_index = { 0 };

        auto actual_branches = gpu::branches(parent_index);
        EXPECT_EQ(expected_branches, actual_branches);

        auto actual_parent_index =
            gpu::tree_reduce(parent_index, actual_branches);
        EXPECT_EQ(expected_parent_index, actual_parent_index);
    }
}
