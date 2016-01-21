#include <array>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "cell_tree.hpp"
#include "swcio.hpp"
#include "json/src/json.hpp"

using json = nlohmann::json;
using range = memory::Range;

TEST(cell_tree, from_parent_index) {
    // tree with single branch corresponding to the root node
    // this is equivalent to a single compartment model
    //      CASE 1 : single root node in parent_index
    {
        std::vector<int> parent_index = {0};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 1);
        EXPECT_EQ(tree.num_children(0), 0);
    }
    //      CASE 2 : empty parent_index
    {
        std::vector<int> parent_index;
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 1);
        EXPECT_EQ(tree.num_children(0), 0);
    }
    // tree with two segments off the root node
    {
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 3);
        // the root has 2 children
        EXPECT_EQ(tree.num_children(0), 2);
        // the children are leaves
        EXPECT_EQ(tree.num_children(1), 0);
        EXPECT_EQ(tree.num_children(2), 0);
    }
    {
        // tree with three segments off the root node
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 4);
        // the root has 3 children
        EXPECT_EQ(tree.num_children(0), 3);
        // the children are leaves
        EXPECT_EQ(tree.num_children(1), 0);
        EXPECT_EQ(tree.num_children(2), 0);
        EXPECT_EQ(tree.num_children(3), 0);
    }
    {
        // tree with three segments off the root node, and another 2 segments off of the third branch from the root node
        std::vector<int> parent_index =
            {0, 0, 1, 2, 0, 4, 0, 6, 7, 8, 9, 8, 11, 12};
        cell_tree tree(parent_index);
        EXPECT_EQ(tree.num_segments(), 6);
        // the root has 3 children
        EXPECT_EQ(tree.num_children(0), 3);
        // one of the chilren has 2 children ...
        EXPECT_EQ(tree.num_children(3), 2);
        // the rest are leaves
        EXPECT_EQ(tree.num_children(1), 0);
        EXPECT_EQ(tree.num_children(2), 0);
        EXPECT_EQ(tree.num_children(4), 0);
        EXPECT_EQ(tree.num_children(5), 0);
    }
    {
        //
        //              0
        //             /
        //            1
        //           / \
        //          2   3
        std::vector<int> parent_index = {0,0,1,1};
        cell_tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 4);

        EXPECT_EQ(tree.num_children(0), 1);
        EXPECT_EQ(tree.num_children(1), 2);
        EXPECT_EQ(tree.num_children(2), 0);
        EXPECT_EQ(tree.num_children(3), 0);
    }
    {
        //
        //              0
        //             / \
        //            1   2
        //           / \
        //          3   4
        std::vector<int> parent_index = {0,0,0,1,1};
        cell_tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 5);

        EXPECT_EQ(tree.num_children(0), 2);
        EXPECT_EQ(tree.num_children(1), 2);
        EXPECT_EQ(tree.num_children(2), 0);
        EXPECT_EQ(tree.num_children(3), 0);
        EXPECT_EQ(tree.num_children(4), 0);
    }
    {
        //              0
        //             / \
        //            1   2
        //           / \
        //          3   4
        //             / \
        //            5   6
        std::vector<int> parent_index = {0,0,0,1,1,4,4};
        cell_tree tree(parent_index);

        EXPECT_EQ(tree.num_segments(), 7);

        EXPECT_EQ(tree.num_children(0), 2);
        EXPECT_EQ(tree.num_children(1), 2);
        EXPECT_EQ(tree.num_children(2), 0);
        EXPECT_EQ(tree.num_children(3), 0);
        EXPECT_EQ(tree.num_children(4), 2);
        EXPECT_EQ(tree.num_children(5), 0);
        EXPECT_EQ(tree.num_children(6), 0);
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
        std::vector<int> parent_index = {0,0,0};
        tree t;
        t.init_from_parent_index(parent_index);
        t.change_root(1);

        EXPECT_EQ(t.num_nodes(), 3);

        EXPECT_EQ(t.num_children(0), 1);
        EXPECT_EQ(t.num_children(1), 1);
        EXPECT_EQ(t.num_children(2), 0);
    }
    {
        // a cell with the following structure
        // make 1 the new root
        //              0          0
        //             / \        /|\
        //            1   2  ->  1 2 3
        //           / \             |
        //          3   4            4
        std::vector<int> parent_index = {0,0,0,1,1};
        tree t;
        t.init_from_parent_index(parent_index);
        t.change_root(1);

        EXPECT_EQ(t.num_nodes(), 5);

        EXPECT_EQ(t.num_children(0), 3);
        EXPECT_EQ(t.num_children(1), 0);
        EXPECT_EQ(t.num_children(2), 0);
        EXPECT_EQ(t.num_children(3), 1);
        EXPECT_EQ(t.num_children(4), 0);
    }
    {
        // a cell with the following structure
        // make 1 the new root
        // unlike earlier tests, this decreases the depth
        // of the tree
        //              0         0
        //             / \       /|\
        //            1   2 ->  1 2 5
        //           / \         / \ \
        //          3   4       3   4 6
        //             / \
        //            5   6
        std::vector<int> parent_index = {0,0,0,1,1,4,4};
        tree t;
        t.init_from_parent_index(parent_index);

        t.change_root(1);

        EXPECT_EQ(t.num_nodes(), 7);

        EXPECT_EQ(t.num_children(0), 3);
        EXPECT_EQ(t.num_children(1), 0);
        EXPECT_EQ(t.num_children(2), 2);
        EXPECT_EQ(t.num_children(3), 0);
        EXPECT_EQ(t.num_children(4), 0);
        EXPECT_EQ(t.num_children(5), 1);
        EXPECT_EQ(t.num_children(6), 0);
    }
}

TEST(cell_tree, balance) {
    {
        // a cell with the following structure
        // will balance around 1
        //              0         0
        //             / \       /|\
        //            1   2 ->  1 2 5
        //           / \         / \ \
        //          3   4       3   4 6
        //             / \
        //            5   6
        std::vector<int> parent_index = {0,0,0,1,1,4,4};
        cell_tree t(parent_index);

        t.balance();

        // the soma (original root) has moved to 5 in the new tree
        EXPECT_EQ(t.soma(), 5);

        EXPECT_EQ(t.num_segments(), 7);
        EXPECT_EQ(t.num_children(0),3);
        EXPECT_EQ(t.num_children(1),0);
        EXPECT_EQ(t.num_children(2),2);
        EXPECT_EQ(t.num_children(3),0);
        EXPECT_EQ(t.num_children(4),0);
        EXPECT_EQ(t.num_children(5),1);
        EXPECT_EQ(t.num_children(6),0);
        EXPECT_EQ(t.parent(0),-1);
        EXPECT_EQ(t.parent(1), 0);
        EXPECT_EQ(t.parent(2), 0);
        EXPECT_EQ(t.parent(3), 0);
        EXPECT_EQ(t.parent(4), 2);
        EXPECT_EQ(t.parent(5), 2);
        EXPECT_EQ(t.parent(6), 5);

        t.to_graphviz("cell.dot");
    }
}

// this test doesn't test anything yet... it just loads each cell in turn
// from a json file and creates a .dot file for it
TEST(cell_tree, json_load) {
    json  cell_data;
    std::ifstream("cells_small.json") >> cell_data;

    for(auto c : range(0,cell_data.size())) {
        std::vector<int> parent_index = cell_data[c]["parent_index"];
        cell_tree tree(parent_index);
        //tree.to_graphviz("cell_" + std::to_string(c) + ".dot");
        tree.to_graphviz("cell" + std::to_string(c) + ".dot");
    }
}

// SWC tests
void expect_cell_equals(const neuron::io::cell_record &expected,
                        const neuron::io::cell_record &actual)
{
    EXPECT_EQ(expected.id(), actual.id());
    EXPECT_EQ(expected.type(), actual.type());
    EXPECT_FLOAT_EQ(expected.x(), actual.x());
    EXPECT_FLOAT_EQ(expected.y(), actual.y());
    EXPECT_FLOAT_EQ(expected.z(), actual.z());
    EXPECT_FLOAT_EQ(expected.radius(), actual.radius());
    EXPECT_EQ(expected.parent(), actual.parent());
}

TEST(cell_record, construction)
{
    using namespace neuron::io;

    {
        // force an invalid type
        cell_record::kind invalid_type = static_cast<cell_record::kind>(100);
        EXPECT_THROW(cell_record cell(invalid_type, 7, 1., 1., 1., 1., 5),
                     std::invalid_argument);
    }

    {
        // invalid id
        EXPECT_THROW(cell_record cell(
                         cell_record::custom, -3, 1., 1., 1., 1., 5),
                     std::invalid_argument);
    }

    {
        // invalid parent id
        EXPECT_THROW(cell_record cell(
                         cell_record::custom, 0, 1., 1., 1., 1., -5),
                     std::invalid_argument);
    }

    {
        // invalid radius
        EXPECT_THROW(cell_record cell(
                         cell_record::custom, 0, 1., 1., 1., -1., -1),
                     std::invalid_argument);
    }

    {
        // parent_id > id
        EXPECT_THROW(cell_record cell(
                         cell_record::custom, 0, 1., 1., 1., 1., 2),
                     std::invalid_argument);
    }

    {
        // parent_id == id
        EXPECT_THROW(cell_record cell(
                         cell_record::custom, 0, 1., 1., 1., 1., 0),
                     std::invalid_argument);
    }

    {
        // check standard construction by value
        cell_record cell(cell_record::custom, 0, 1., 1., 1., 1., -1);
        EXPECT_EQ(cell.id(), 0);
        EXPECT_EQ(cell.type(), cell_record::custom);
        EXPECT_EQ(cell.x(), 1.);
        EXPECT_EQ(cell.y(), 1.);
        EXPECT_EQ(cell.z(), 1.);
        EXPECT_EQ(cell.radius(), 1.);
        EXPECT_EQ(cell.diameter(), 2*1.);
        EXPECT_EQ(cell.parent(), -1);
    }

    {
        // check copy constructor
        cell_record cell_orig(cell_record::custom, 0, 1., 1., 1., 1., -1);
        cell_record cell(cell_orig);
        expect_cell_equals(cell_orig, cell);
    }
}

TEST(swc_parser, invalid_input)
{
    using namespace neuron::io;

    {
        // check incomplete lines; missing parent
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830\n");
        cell_record cell;
        EXPECT_THROW(is >> cell, std::logic_error);
    }

    {
        // Check long lines
        std::istringstream is(std::string(256, 'a') + "\n");
        cell_record cell;
        EXPECT_THROW(is >> cell, std::runtime_error);
    }

    {
        // Check non-parsable values
        std::istringstream is("1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        cell_record cell;
        EXPECT_THROW(is >> cell, std::logic_error);
    }

    {
        // Check invalid cell value
        std::istringstream is("1 10 14.566132 34.873772 7.857000 0.717830 -1\n");
        cell_record cell;
        EXPECT_THROW(is >> cell, std::invalid_argument);
    }
}


TEST(swc_parser, valid_input)
{
    using namespace neuron::io;

    {
        // check empty file; no cell may be parsed
        cell_record cell, cell_orig;
        std::istringstream is("");
        EXPECT_NO_THROW(is >> cell);
        expect_cell_equals(cell_orig, cell);
    }

    {
        // check comment-only file not ending with a newline;
        // no cell may be parsed
        cell_record cell, cell_orig;
        std::istringstream is("#comment\n#comment");
        EXPECT_NO_THROW(is >> cell);
        expect_cell_equals(cell_orig, cell);
    }


    {
        // check last line case (no newline at the end)
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830 -1");
        cell_record cell;
        EXPECT_NO_THROW(is >> cell);
        EXPECT_EQ(0, cell.id());    // zero-based indexing
        EXPECT_EQ(cell_record::soma, cell.type());
        EXPECT_FLOAT_EQ(14.566132, cell.x());
        EXPECT_FLOAT_EQ(34.873772, cell.y());
        EXPECT_FLOAT_EQ( 7.857000, cell.z());
        EXPECT_FLOAT_EQ( 0.717830, cell.radius());
        EXPECT_FLOAT_EQ( -1, cell.parent());
    }

    {
        // check valid input with a series of records
        std::vector<cell_record> cells_orig = {
            cell_record(cell_record::soma, 0,
                        14.566132, 34.873772, 7.857000, 0.717830, -1),
            cell_record(cell_record::dendrite, 1,
                        14.566132+1, 34.873772+1, 7.857000+1, 0.717830+1, -1)
        };

        std::stringstream swc_input;
        swc_input << "# this is a comment\n";
        swc_input << "# this is a comment\n";
        for (auto c : cells_orig)
            swc_input << c << "\n";

        swc_input << "# this is a final comment\n";
        try {
            std::size_t nr_records = 0;
            cell_record cell;
            while ( !(swc_input >> cell).eof()) {
                ASSERT_LT(nr_records, cells_orig.size());
                expect_cell_equals(cells_orig[nr_records], cell);
                ++nr_records;
            }
        } catch (std::exception &e) {
            ADD_FAILURE();
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
