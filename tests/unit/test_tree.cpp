#include <fstream>
#include <iostream>
#include <json/json.hpp>
#include <numeric>
#include <vector>

#include "../gtest.h"

#include <cell_tree.hpp>
#include <util/debug.hpp>

// Path to data directory can be overriden at compile time.
#if !defined(DATADIR)
#define DATADIR "../data"
#endif

using json = nlohmann::json;

using namespace nest::mc;
using int_type = cell_tree::int_type;


TEST(cell_tree, from_parent_index) {
    auto no_parent = cell_tree::no_parent;

    // tree with single branch corresponding to the root node
    // this is equivalent to a single compartment model
    //      CASE 1 : single root node in parent_index
    {
        std::vector<int_type> parent_index = {0};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 1u);
        EXPECT_EQ(tree.num_children(0), 0u);
    }
    //      CASE 2 : empty parent_index
    {
        std::vector<int_type> parent_index;
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 1u);
        EXPECT_EQ(tree.num_children(0), 0u);
    }

    {
        //
        //        0               0
        //       / \             / \.
        //      1   4      =>   1   2
        //     /     \.
        //    2       5
        //   /
        //  3
        //
        std::vector<int_type> parent_index =
            {0, 0, 1, 2, 0, 4};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 3u);
        // the root has 2 children
        EXPECT_EQ(tree.num_children(0), 2u);
        // the children are leaves
        EXPECT_EQ(tree.num_children(1), 0u);
        EXPECT_EQ(tree.num_children(2), 0u);
    }
    {
        //
        //        0               0
        //       /|\             /|\.
        //      1 4 6      =>   1 2 3
        //     /  |  \.
        //    2   5   7
        //   /         \.
        //  3           8
        //
        std::vector<int_type> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8};

        cell_tree tree(parent_index);
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
        //        0               0
        //       /|\             /|\.
        //      1 4 6      =>   1 2 3
        //     /  |  \             / \.
        //    2   5   7           4   5
        //   /         \.
        //  3           8
        //             / \.
        //            9   11
        //           /     \.
        //          10     12
        //                   \.
        //                   13
        //
        std::vector<int_type> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12};
        cell_tree tree(parent_index);
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
        cell_tree tree(parent_index);

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
        cell_tree tree(parent_index);

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
        cell_tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 7u);

        EXPECT_EQ(tree.num_children(0), 2u);
        EXPECT_EQ(tree.num_children(1), 2u);
        EXPECT_EQ(tree.num_children(2), 0u);
        EXPECT_EQ(tree.num_children(3), 0u);
        EXPECT_EQ(tree.num_children(4), 2u);
        EXPECT_EQ(tree.num_children(5), 0u);
        EXPECT_EQ(tree.num_children(6), 0u);
    }
}

TEST(cell_tree, depth_from_root) {
    {
        //              0
        //             / \.
        //            1   2
        //           / \.
        //          3   4
        //             / \.
        //            5   6
        std::vector<int_type> parent_index = {0,0,0,1,1,4,4};
        cell_tree tree(parent_index);
        auto depth = tree.depth_from_root();

        EXPECT_EQ(depth[0], 0u);
        EXPECT_EQ(depth[1], 1u);
        EXPECT_EQ(depth[2], 1u);
        EXPECT_EQ(depth[3], 2u);
        EXPECT_EQ(depth[4], 2u);
        EXPECT_EQ(depth[5], 3u);
        EXPECT_EQ(depth[6], 3u);
    }
    {
        //              0
        //             / \.
        //            1   2
        //               / \.
        //              3   4
        //                 / \.
        //                5   6
        std::vector<int_type> parent_index = {0,0,0,2,2,4,4};
        cell_tree tree(parent_index);
        auto depth = tree.depth_from_root();

        EXPECT_EQ(depth[0], 0u);
        EXPECT_EQ(depth[1], 1u);
        EXPECT_EQ(depth[2], 1u);
        EXPECT_EQ(depth[3], 2u);
        EXPECT_EQ(depth[4], 2u);
        EXPECT_EQ(depth[5], 3u);
        EXPECT_EQ(depth[6], 3u);
    }
}

TEST(tree, change_root) {
    {
        // a cell with the following structure
        // make 1 the new root
        //              0       0
        //             / \      |
        //            1   2 ->  1
        //                      |
        //                      2
        std::vector<int_type> parent_index = {0,0,0};
        tree<int_type> t(parent_index);
        t.change_root(1);

        EXPECT_EQ(t.num_nodes(), 3u);

        EXPECT_EQ(t.num_children(0), 1u);
        EXPECT_EQ(t.num_children(1), 1u);
        EXPECT_EQ(t.num_children(2), 0u);
    }
    {
        // a cell with the following structure
        // make 1 the new root
        //              0          0
        //             / \        /|\.
        //            1   2  ->  1 2 3
        //           / \             |
        //          3   4            4
        std::vector<int_type> parent_index = {0,0,0,1,1};
        tree<int_type> t(parent_index);
        t.change_root(1u);

        EXPECT_EQ(t.num_nodes(), 5u);

        EXPECT_EQ(t.num_children(0), 3u);
        EXPECT_EQ(t.num_children(1), 0u);
        EXPECT_EQ(t.num_children(2), 0u);
        EXPECT_EQ(t.num_children(3), 1u);
        EXPECT_EQ(t.num_children(4), 0u);
    }
    {
        // a cell with the following structure
        // make 1 the new root
        // unlike earlier tests, this decreases the depth
        // of the tree
        //              0         0
        //             / \       /|\.
        //            1   2 ->  1 2 5
        //           / \         / \ \.
        //          3   4       3   4 6
        //             / \.
        //            5   6
        std::vector<int_type> parent_index = {0,0,0,1,1,4,4};
        tree<int_type> t(parent_index);

        t.change_root(1);

        EXPECT_EQ(t.num_nodes(), 7u);

        EXPECT_EQ(t.num_children(0), 3u);
        EXPECT_EQ(t.num_children(1), 0u);
        EXPECT_EQ(t.num_children(2), 2u);
        EXPECT_EQ(t.num_children(3), 0u);
        EXPECT_EQ(t.num_children(4), 0u);
        EXPECT_EQ(t.num_children(5), 1u);
        EXPECT_EQ(t.num_children(6), 0u);
    }
}

TEST(cell_tree, balance) {
    auto no_parent = cell_tree::no_parent;

    {
        // a cell with the following structure
        // will balance around 1
        //              0         0
        //             / \       /|\.
        //            1   2 ->  1 2 5
        //           / \         / \ \.
        //          3   4       3   4 6
        //             / \.
        //            5   6
        std::vector<int_type> parent_index = {0,0,0,1,1,4,4};
        cell_tree t(parent_index);

        t.balance();

        // the soma (original root) has moved to 5 in the new tree
        EXPECT_EQ(t.soma(), 5u);

        EXPECT_EQ(t.num_segments(), 7u);
        EXPECT_EQ(t.num_children(0),3u);
        EXPECT_EQ(t.num_children(1),0u);
        EXPECT_EQ(t.num_children(2),2u);
        EXPECT_EQ(t.num_children(3),0u);
        EXPECT_EQ(t.num_children(4),0u);
        EXPECT_EQ(t.num_children(5),1u);
        EXPECT_EQ(t.num_children(6),0u);
        EXPECT_EQ(t.parent(0), no_parent);
        EXPECT_EQ(t.parent(1), 0u);
        EXPECT_EQ(t.parent(2), 0u);
        EXPECT_EQ(t.parent(3), 0u);
        EXPECT_EQ(t.parent(4), 2u);
        EXPECT_EQ(t.parent(5), 2u);
        EXPECT_EQ(t.parent(6), 5u);

        //t.to_graphviz("cell.dot");
    }
}

// this test doesn't test anything yet... it just loads each cell in turn
// from a json file and creates a .dot file for it
TEST(cell_tree, json_load)
{
    json  cell_data;
    std::string path{DATADIR};

    path += "/cells_small.json";
    std::ifstream(path) >> cell_data;

    for(auto c : util::make_span(0,cell_data.size())) {
        std::vector<int_type> parent_index = cell_data[c]["parent_index"];
        cell_tree tree(parent_index);
        //tree.to_graphviz("cell" + std::to_string(c) + ".dot");
    }
}

TEST(tree, make_parent_index)
{
    /*
    // just the soma
    {
        std::vector<int> parent_index = {0};
        std::vector<int> counts = {1};
        nest::mc::tree t(parent_index);
        auto new_parent_index = make_parent_index(t, counts);
        EXPECT_EQ(parent_index.size(), new_parent_index.size());
    }
    // just a soma with 5 compartments
    {
        std::vector<int> parent_index = {0};
        std::vector<int> counts = {5};
        nest::mc::tree t(parent_index);
        auto new_parent_index = make_parent_index(t, counts);
        EXPECT_EQ(new_parent_index.size(), (unsigned)counts[0]);
        EXPECT_EQ(new_parent_index[0], 0);
        for(auto i=1u; i<new_parent_index.size(); ++i) {
            EXPECT_EQ((unsigned)new_parent_index[i], i-1);
        }
    }
    // some trees with single compartment per segment
    {
        auto trees = {
            // 0
            // |
            // 1
            std::vector<int>{0,0},
            //          0
            //         / \.
            //        1   2
            std::vector<int>{0,0,0},
            //          0
            //         / \.
            //        1   4
            //       / \  |\.
            //      2   3 5 6
            std::vector<int>{0,0,0,1,1,2,2}
        };
        for(auto &parent_index : trees) {
            std::vector<int> counts(parent_index.size(), 1);
            nest::mc::tree t(parent_index);
            auto new_parent_index = make_parent_index(t, counts);
            EXPECT_EQ(parent_index, new_parent_index);
        }
    }
    // a tree with multiple compartments per segment
    //
    //              0
    //             / \.
    //            1   8
    //           /     \.
    //          2       9
    //         /.
    //        3
    //       / \.
    //      4   6
    //     /     \.
    //    5       7
    {
        std::vector<int> parent_index = {0,0,1,2,3,4,3,6,0,8};
        std::vector<int> counts = {1,3,2,2,2};
        nest::mc::tree t(parent_index);
        auto new_parent_index = make_parent_index(t, counts);
        EXPECT_EQ(parent_index, new_parent_index);
    }
    */
}
