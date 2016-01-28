#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest.h"

#include <swcio.hpp>

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
        // Check non-parsable values
        std::istringstream is("1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        cell_record cell;
        EXPECT_THROW(is >> cell, std::logic_error);
    }

    {
        // Check invalid cell type
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

TEST(swc_parser, from_allen_db)
{
    using namespace neuron;

    auto fname = "../data/example.swc";
    std::ifstream fid(fname);
    if(!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // load the cell records into a std::vector
    std::vector<io::cell_record> nodes;
    io::cell_record node;
    while( !(fid >> node).eof()) {
        nodes.push_back(std::move(node));
    }
    // verify that the correct number of nodes was read
    EXPECT_EQ(nodes.size(), 1058u);
}
