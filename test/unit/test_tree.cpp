#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "../gtest.h"

#include <tree.hpp>

using namespace arb;
using int_type = tree::int_type;
using iarray = tree::iarray;

TEST(tree, from_segment_index) {
    auto no_parent = tree::no_parent;

    // tree with single branch corresponding to the root node
    // this is equivalent to a single compartment model
    //      CASE 1 : single root node in parent_index
    {
        std::vector<int_type> parent_index = {0};
        tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 1u);
        EXPECT_EQ(tree.num_children(0), 0u);
    }

    {
        //
        //     0
        //    / \.
        //   1   2
        //
        std::vector<int_type> parent_index = {0, 0, 0};
        tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 3u);
        // the root has 2 children
        EXPECT_EQ(tree.num_children(0), 2u);
        // the children are leaves
        EXPECT_EQ(tree.num_children(1), 0u);
        EXPECT_EQ(tree.num_children(2), 0u);
    }
    {
        //
        //     0-1-2-3
        //
        std::vector<int_type> parent_index = {0, 0, 1, 2};
        tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 4u);
        // all non-leaf nodes have 1 child
        EXPECT_EQ(tree.num_children(0), 1u);
        EXPECT_EQ(tree.num_children(1), 1u);
        EXPECT_EQ(tree.num_children(2), 1u);
        EXPECT_EQ(tree.num_children(3), 0u);
    }
    {
        //
        //     0
        //    /|\.
        //   1 2 3
        //
        std::vector<int_type> parent_index = {0, 0, 0, 0};
        tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 4u);
        // the root has 3 children
        EXPECT_EQ(tree.num_children(0), 3u);
        // the children are leaves
        EXPECT_EQ(tree.num_children(1), 0u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(3), 0u);

        // Check new structure
        EXPECT_EQ(no_parent, tree.parent(0));
        EXPECT_EQ(0u, tree.parent(1));
        EXPECT_EQ(0u, tree.parent(2));
        EXPECT_EQ(0u, tree.parent(3));
    }
    {
        //
        //   0
        //  /|\.
        // 1 2 3
        //    / \.
        //   4   5
        //
        std::vector<int_type> parent_index = {0, 0, 0, 0, 3, 3};
        tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 6u);
        // the root has 3 children
        EXPECT_EQ(tree.num_children(0), 3u);
        // one of the chilren has 2 children ...
        EXPECT_EQ(tree.num_children(3), 2u);
        // the rest are leaves
        EXPECT_EQ(tree.num_children(1), 0u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(4), 0u);
        EXPECT_EQ(tree.num_children(5), 0u);

        // Check new structure
        EXPECT_EQ(no_parent, tree.parent(0));
        EXPECT_EQ(0u, tree.parent(1));
        EXPECT_EQ(0u, tree.parent(2));
        EXPECT_EQ(0u, tree.parent(3));
        EXPECT_EQ(3u, tree.parent(4));
        EXPECT_EQ(3u, tree.parent(5));
    }
    {
        //
        //              0
        //             /
        //            1
        //           / \.
        //          2   3
        std::vector<int_type> parent_index = {0,0,1,1};
        tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 4u);

        EXPECT_EQ(tree.num_children(0), 1u);
        EXPECT_EQ(tree.num_children(1), 2u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(3), 0u);
    }
    {
        //
        //              0
        //             /|\.
        //            1 4 5
        //           / \.
        //          2   3
        std::vector<int_type> parent_index = {0,0,1,1,0,0};
        tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 6u);

        EXPECT_EQ(tree.num_children(0), 3u);
        EXPECT_EQ(tree.num_children(1), 2u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(3), 0u);
        EXPECT_EQ(tree.num_children(4), 0u);

        // Check children
        EXPECT_EQ(1u, tree.children(0)[0]);
        EXPECT_EQ(4u, tree.children(0)[1]);
        EXPECT_EQ(5u, tree.children(0)[2]);
        EXPECT_EQ(2u, tree.children(1)[0]);
        EXPECT_EQ(3u, tree.children(1)[1]);
    }
    {
        //              0
        //             / \.
        //            1   2
        //           / \.
        //          3   4
        //             / \.
        //            5   6
        std::vector<int_type> parent_index = {0,0,0,1,1,4,4};
        tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 7u);

        EXPECT_EQ(tree.num_children(0), 2u);
        EXPECT_EQ(tree.num_children(1), 2u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(3), 0u);
        EXPECT_EQ(tree.num_children(4), 2u);
        EXPECT_EQ(tree.num_children(5), 0u);
        EXPECT_EQ(tree.num_children(6), 0u);

        EXPECT_EQ(tree.children(0)[0], 1u);
        EXPECT_EQ(tree.children(0)[1], 2u);
        EXPECT_EQ(tree.children(1)[0], 3u);
        EXPECT_EQ(tree.children(1)[1], 4u);
        EXPECT_EQ(tree.children(4)[0], 5u);
        EXPECT_EQ(tree.children(4)[1], 6u);
    }
}

TEST(tree, depth_from_root) {
    // tree with single branch corresponding to the root node
    // this is equivalent to a single compartment model
    //      CASE 1 : single root node in parent_index
    {
        std::vector<int_type> parent_index = {0};
        iarray expected = {0u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }

    {
        //     0
        //    / \.
        //   1   2
        std::vector<int_type> parent_index = {0, 0, 0};
        iarray expected = {0u, 1u, 1u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //     0-1-2-3
        std::vector<int_type> parent_index = {0, 0, 1, 2};
        iarray expected = {0u, 1u, 2u, 3u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //
        //     0
        //    /|\.
        //   1 2 3
        //
        std::vector<int_type> parent_index = {0, 0, 0, 0};
        iarray expected = {0u, 1u, 1u, 1u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //
        //   0
        //  /|\.
        // 1 2 3
        //    / \.
        //   4   5
        //
        std::vector<int_type> parent_index = {0, 0, 0, 0, 3, 3};
        iarray expected = {0u, 1u, 1u, 1u, 2u, 2u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //
        //              0
        //             /
        //            1
        //           / \.
        //          2   3
        std::vector<int_type> parent_index = {0,0,1,1};
        iarray expected = {0u, 1u, 2u, 2u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //
        //              0
        //             /|\.
        //            1 4 5
        //           / \.
        //          2   3
        std::vector<int_type> parent_index = {0,0,1,1,0,0};
        iarray expected = {0u, 1u, 2u, 2u, 1u, 1u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
    {
        //              0
        //             / \.
        //            1   2
        //           / \.
        //          3   4
        //             / \.
        //            5   6
        std::vector<int_type> parent_index = {0,0,0,1,1,4,4};
        iarray expected = {0u, 1u, 1u, 2u, 2u, 3u, 3u};
        EXPECT_EQ(expected, depth_from_root(tree(parent_index)));
    }
}

