#include <array>
#include <exception>
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <vector>

#include "gtest.h"

#include <cell.hpp>
#include <swcio.hpp>

// SWC tests
void expect_record_equals(const nest::mc::io::swc_record &expected,
                          const nest::mc::io::swc_record &actual)
{
    EXPECT_EQ(expected.id(), actual.id());
    EXPECT_EQ(expected.type(), actual.type());
    EXPECT_FLOAT_EQ(expected.x(), actual.x());
    EXPECT_FLOAT_EQ(expected.y(), actual.y());
    EXPECT_FLOAT_EQ(expected.z(), actual.z());
    EXPECT_FLOAT_EQ(expected.radius(), actual.radius());
    EXPECT_EQ(expected.parent(), actual.parent());
}

TEST(swc_record, construction)
{
    using namespace nest::mc::io;

    {
        // force an invalid type
        swc_record::kind invalid_type = static_cast<swc_record::kind>(100);
        EXPECT_THROW(swc_record record(invalid_type, 7, 1., 1., 1., 1., 5),
                     std::invalid_argument);
    }

    {
        // invalid id
        EXPECT_THROW(swc_record record(
                         swc_record::kind::custom, -3, 1., 1., 1., 1., 5),
                     std::invalid_argument);
    }

    {
        // invalid parent id
        EXPECT_THROW(swc_record record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., -5),
                     std::invalid_argument);
    }

    {
        // invalid radius
        EXPECT_THROW(swc_record record(
                         swc_record::kind::custom, 0, 1., 1., 1., -1., -1),
                     std::invalid_argument);
    }

    {
        // parent_id > id
        EXPECT_THROW(swc_record record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., 2),
                     std::invalid_argument);
    }

    {
        // parent_id == id
        EXPECT_THROW(swc_record record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., 0),
                     std::invalid_argument);
    }

    {
        // check standard construction by value
        swc_record record(swc_record::kind::custom, 0, 1., 1., 1., 1., -1);
        EXPECT_EQ(record.id(), 0);
        EXPECT_EQ(record.type(), swc_record::kind::custom);
        EXPECT_EQ(record.x(), 1.);
        EXPECT_EQ(record.y(), 1.);
        EXPECT_EQ(record.z(), 1.);
        EXPECT_EQ(record.radius(), 1.);
        EXPECT_EQ(record.diameter(), 2*1.);
        EXPECT_EQ(record.parent(), -1);
    }

    {
        // check copy constructor
        swc_record record_orig(swc_record::kind::custom, 0, 1., 1., 1., 1., -1);
        swc_record record(record_orig);
        expect_record_equals(record_orig, record);
    }
}

TEST(swc_record, comparison)
{
    using namespace nest::mc::io;

    {
        // check comparison operators
        swc_record record0(swc_record::kind::custom, 0, 1., 1., 1., 1., -1);
        swc_record record1(swc_record::kind::custom, 0, 2., 3., 4., 5., -1);
        swc_record record2(swc_record::kind::custom, 1, 2., 3., 4., 5., -1);
        EXPECT_EQ(record0, record1);
        EXPECT_LT(record0, record2);
        EXPECT_GT(record2, record1);
    }

}

TEST(swc_parser, invalid_input)
{
    using namespace nest::mc::io;

    {
        // check incomplete lines; missing parent
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830\n");
        swc_record record;
        EXPECT_THROW(is >> record, swc_parse_error);
    }

    {
        // Check non-parsable values
        std::istringstream is(
            "1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        swc_record record;
        EXPECT_THROW(is >> record, swc_parse_error);
    }

    {
        // Check invalid record type
        std::istringstream is(
            "1 10 14.566132 34.873772 7.857000 0.717830 -1\n");
        swc_record record;
        EXPECT_THROW(is >> record, swc_parse_error);
    }

    {
        // Non-contiguous numbering in branches is considered invalid
        //        1
        //       / \.
        //      2   3
        //     /
        //    4
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 1 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 1 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 1 14.566132 34.873772 7.857000 0.717830 2\n";

        std::vector<swc_record> records;
        try {
            for (auto c : swc_get_records<swc_io_clean>(is)) {
                records.push_back(std::move(c));
            }

            FAIL() << "expected swc_parse_error, none was thrown\n";
        } catch (const swc_parse_error &e) {
            SUCCEED();
        }
    }
}


TEST(swc_parser, valid_input)
{
    using namespace nest::mc::io;

    {
        // check empty file; no record may be parsed
        swc_record record, record_orig;
        std::istringstream is("");
        EXPECT_NO_THROW(is >> record);
        expect_record_equals(record_orig, record);
    }

    {
        // check comment-only file not ending with a newline;
        // no record may be parsed
        swc_record record, record_orig;
        std::istringstream is("#comment\n#comment");
        EXPECT_NO_THROW(is >> record);
        expect_record_equals(record_orig, record);
    }


    {
        // check last line case (no newline at the end)
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830 -1");
        swc_record record;
        EXPECT_NO_THROW(is >> record);
        EXPECT_EQ(0, record.id());    // zero-based indexing
        EXPECT_EQ(swc_record::kind::soma, record.type());
        EXPECT_FLOAT_EQ(14.566132, record.x());
        EXPECT_FLOAT_EQ(34.873772, record.y());
        EXPECT_FLOAT_EQ( 7.857000, record.z());
        EXPECT_FLOAT_EQ( 0.717830, record.radius());
        EXPECT_FLOAT_EQ( -1, record.parent());
    }

    {
        // check valid input with a series of records
        std::vector<swc_record> records_orig = {
            swc_record(swc_record::kind::soma, 0,
                        14.566132, 34.873772, 7.857000, 0.717830, -1),
            swc_record(swc_record::kind::dendrite, 1,
                        14.566132+1, 34.873772+1, 7.857000+1, 0.717830+1, -1)
        };

        std::stringstream swc_input;
        swc_input << "# this is a comment\n";
        swc_input << "# this is a comment\n";
        for (auto c : records_orig)
            swc_input << c << "\n";

        swc_input << "# this is a final comment\n";

        std::size_t nr_records = 0;
        for (auto record : swc_get_records<swc_io_raw>(swc_input)) {
            ASSERT_LT(nr_records, records_orig.size());
            expect_record_equals(records_orig[nr_records], record);
            ++nr_records;
        }
    }
}

TEST(swc_parser, from_allen_db)
{
    using namespace nest::mc;

    auto fname = "../data/example.swc";
    std::ifstream fid(fname);
    if(!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // load the record records into a std::vector
    std::vector<io::swc_record> nodes;
    for (auto node : io::swc_get_records<io::swc_io_raw>(fid)) {
        nodes.push_back(std::move(node));
    }

    // verify that the correct number of nodes was read
    EXPECT_EQ(nodes.size(), 1058u);
}

TEST(swc_parser, input_cleaning)
{
    using namespace nest::mc::io;

    {
        // Check duplicates
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";

        EXPECT_EQ(2u, swc_get_records<swc_io_clean>(is).size());
    }

    {
        // Check multiple trees
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        auto records = swc_get_records<swc_io_clean>(is);
        EXPECT_EQ(2u, records.size());
    }

    {
        // Check unsorted input
        std::stringstream is;
        is << "3 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";

        std::array<swc_record::id_type, 4> expected_id_list = {{ 0, 1, 2, 3 }};

        auto expected_id = expected_id_list.cbegin();
        for (auto c : swc_get_records<swc_io_clean>(is)) {
            EXPECT_EQ(*expected_id, c.id());
            ++expected_id;
        }

        // Check that we have read through the whole input
        EXPECT_EQ(expected_id_list.end(), expected_id);
    }

    {
        // Check holes in numbering
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "21 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "31 2 14.566132 34.873772 7.857000 0.717830 21\n";
        is << "41 2 14.566132 34.873772 7.857000 0.717830 21\n";
        is << "51 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "61 2 14.566132 34.873772 7.857000 0.717830 51\n";

        std::array<swc_record::id_type, 6> expected_id_list =
            {{ 0, 1, 2, 3, 4, 5 }};
        std::array<swc_record::id_type, 6> expected_parent_list =
            {{ -1, 0, 1, 1, 0, 4 }};

        auto expected_id = expected_id_list.cbegin();
        auto expected_parent = expected_parent_list.cbegin();
        for (auto c : swc_get_records<swc_io_clean>(is)) {
            EXPECT_EQ(*expected_id, c.id());
            EXPECT_EQ(*expected_parent, c.parent());
            ++expected_id;
            ++expected_parent;
        }

        // Check that we have read through the whole input
        EXPECT_EQ(expected_id_list.end(), expected_id);
        EXPECT_EQ(expected_parent_list.end(), expected_parent);
    }
}

TEST(swc_record_ranges, raw)
{
    using namespace nest::mc::io;

    {
        // Check valid usage
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        std::vector<swc_record> records;
        for (auto c : swc_get_records<swc_io_raw>(is)) {
            records.push_back(c);
        }

        EXPECT_EQ(4u, records.size());

        bool entered = false;
        auto citer = records.begin();
        for (auto c : swc_get_records<swc_io_raw>(is)) {
            expect_record_equals(c, *citer++);
            entered = true;
        }

        EXPECT_TRUE(entered);
    }

    {
        // Check out of bounds reads
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";

        auto ibegin = swc_get_records<swc_io_raw>(is).begin();

        EXPECT_NO_THROW(++ibegin);
        EXPECT_THROW(*ibegin, std::out_of_range);

    }

    {
        // Check iterator increments
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";

        auto iter = swc_get_records<swc_io_raw>(is).begin();
        auto iend = swc_get_records<swc_io_raw>(is).end();

        swc_record c;
        EXPECT_NO_THROW(c = *iter++);
        EXPECT_EQ(-1, c.parent());
        EXPECT_EQ(iend, iter);

        // Try to read past eof
        EXPECT_THROW(*iter, std::out_of_range);
    }

    {
        // Check parse error context
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 10 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        std::vector<swc_record> records;
        try {
            for (auto c : swc_get_records<swc_io_raw>(is)) {
                records.push_back(c);
            }

            ADD_FAILURE() << "expected an exception\n";
        } catch (const swc_parse_error &e) {
            EXPECT_EQ(3u, e.lineno());
        }
    }

    {
        // Test empty range
        std::stringstream is("");
        EXPECT_TRUE(swc_get_records<swc_io_raw>(is).empty());
        EXPECT_TRUE(swc_get_records<swc_io_clean>(is).empty());
    }
}

TEST(swc_io, cell_construction)
{
    using namespace nest::mc;

    {
        //
        //    0
        //    |
        //    1
        //    |
        //    2
        //   / \.
        //  3   4
        //       \.
        //        5
        //

        std::stringstream is;
        is << "1 1 0 0 0 2.1 -1\n";
        is << "2 2 0.1 1.2 1.2 1.3 1\n";
        is << "3 2 1.0 2.0 2.2 1.1 2\n";
        is << "4 2 1.5 3.3 1.3 2.2 3\n";
        is << "5 2 2.5 5.3 2.5 0.7 3\n";
        is << "6 2 3.5 2.3 3.7 3.4 5\n";

        using point_type = point<double>;
        std::vector<point_type> points = {
            { 0.0, 0.0, 0.0 },
            { 0.1, 1.2, 1.2 },
            { 1.0, 2.0, 2.2 },
            { 1.5, 3.3, 1.3 },
            { 2.5, 5.3, 2.5 },
            { 3.5, 2.3, 3.7 },
        };

        cell cell = io::swc_read_cell(is);
        EXPECT_TRUE(cell.has_soma());
        EXPECT_EQ(4, cell.num_segments());

        EXPECT_EQ(norm(points[1]-points[2]), cell.cable(1)->length());
        EXPECT_EQ(norm(points[2]-points[3]), cell.cable(2)->length());
        EXPECT_EQ(norm(points[2]-points[4]) + norm(points[4]-points[5]),
                  cell.cable(3)->length());


        // Check each segment separately
        EXPECT_TRUE(cell.segment(0)->is_soma());
        EXPECT_EQ(2.1, cell.soma()->radius());
        EXPECT_EQ(point_type(0, 0, 0), cell.soma()->center());

        for (auto i = 1; i < cell.num_segments(); ++i) {
            EXPECT_TRUE(cell.segment(i)->is_dendrite());
        }

        EXPECT_EQ(1, cell.cable(1)->num_sub_segments());
        EXPECT_EQ(1, cell.cable(2)->num_sub_segments());
        EXPECT_EQ(2, cell.cable(3)->num_sub_segments());


        // Check the radii
        EXPECT_EQ(1.3, cell.cable(1)->radius(0));
        EXPECT_EQ(1.1, cell.cable(1)->radius(1));

        EXPECT_EQ(1.1, cell.cable(2)->radius(0));
        EXPECT_EQ(2.2, cell.cable(2)->radius(1));

        EXPECT_EQ(1.1, cell.cable(3)->radius(0));
        EXPECT_EQ(3.4, cell.cable(3)->radius(1));

        auto len_ratio = norm(points[2]-points[4]) / cell.cable(3)->length();
        EXPECT_NEAR(.7, cell.cable(3)->radius(len_ratio), 1e-15);

        // Double-check radii at joins are equal
        EXPECT_EQ(cell.cable(1)->radius(1),
                  cell.cable(2)->radius(0));

        EXPECT_EQ(cell.cable(1)->radius(1),
                  cell.cable(3)->radius(0));
    }
}

// check that simple ball and stick model with one dendrite attached to a soma
// which is used in the validation tests can be loaded from file and matches
// the one generated with the C++ interface
TEST(swc_parser, from_file_ball_and_stick)
{
    auto fname = "../data/ball_and_stick.swc";
    std::ifstream fid(fname);
    if(!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // read the file into a cell object
    auto cell = nest::mc::io::swc_read_cell(fid);

    // verify that the correct number of nodes was read
    EXPECT_EQ(cell.num_segments(), 2);
    EXPECT_EQ(cell.num_compartments(), 2u);

    // make an equivalent cell via C++ interface
    nest::mc::cell local_cell;
    local_cell.add_soma(6.30785);
    local_cell.add_cable(0, nest::mc::segmentKind::dendrite, 0.5, 0.5, 200);

    EXPECT_TRUE(nest::mc::cell_basic_equality(local_cell, cell));
}

