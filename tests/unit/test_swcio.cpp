#include <array>
#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <vector>

#include "../gtest.h"

#include "../src/cell.hpp"
#include "../src/swcio.hpp"

// Path to data directory can be overriden at compile time.
#if !defined(DATADIR)
#   define DATADIR "../data"
#endif

using namespace nest::mc;

// SWC tests
void expect_record_equals(const io::swc_record& expected,
                          const io::swc_record& actual)
{
    EXPECT_EQ(expected.id, actual.id);
    EXPECT_EQ(expected.type, actual.type);
    EXPECT_FLOAT_EQ(expected.x, actual.x);
    EXPECT_FLOAT_EQ(expected.y, actual.y);
    EXPECT_FLOAT_EQ(expected.z, actual.z);
    EXPECT_FLOAT_EQ(expected.r, actual.r);
    EXPECT_EQ(expected.parent_id, actual.parent_id);
}

TEST(swc_record, construction)
{
    using namespace nest::mc::io;

    {
        // force an invalid type
        swc_record::kind invalid_type = static_cast<swc_record::kind>(100);
        EXPECT_THROW(swc_record(invalid_type, 7, 1., 1., 1., 1., 5).assert_consistent(),
                     swc_error);
    }

    {
        // invalid id
        EXPECT_THROW(swc_record(
                         swc_record::kind::custom, -3, 1., 1., 1., 1., 5).assert_consistent(),
                     swc_error);
    }

    {
        // invalid parent id
        EXPECT_THROW(swc_record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., -5).assert_consistent(),
                     swc_error);
    }

    {
        // invalid radius
        EXPECT_THROW(swc_record(
                         swc_record::kind::custom, 0, 1., 1., 1., -1., -1).assert_consistent(),
                     swc_error);
    }

    {
        // parent_id > id
        EXPECT_THROW(swc_record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., 2).assert_consistent(),
                     swc_error);
    }

    {
        // parent_id == id
        EXPECT_THROW(swc_record(
                         swc_record::kind::custom, 0, 1., 1., 1., 1., 0).assert_consistent(),
                     swc_error);
    }

    {
        // check standard construction by value
        swc_record record(swc_record::kind::custom, 0, 1., 1., 1., 1., -1);
        EXPECT_TRUE(record.is_consistent());
        EXPECT_EQ(record.id, 0);
        EXPECT_EQ(record.type, swc_record::kind::custom);
        EXPECT_EQ(record.x, 1.);
        EXPECT_EQ(record.y, 1.);
        EXPECT_EQ(record.z, 1.);
        EXPECT_EQ(record.r, 1.);
        EXPECT_EQ(record.diameter(), 2*1.);
        EXPECT_EQ(record.parent_id, -1);
    }

    {
        // check copy constructor
        swc_record record_orig(swc_record::kind::custom, 0, 1., 1., 1., 1., -1);
        swc_record record(record_orig);
        expect_record_equals(record_orig, record);
    }
}


TEST(swc_parser, invalid_input_istream)
{
    using namespace nest::mc::io;

    {
        // check incomplete lines; missing parent
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830\n");
        swc_record record;
        is >> record;
        EXPECT_TRUE(is.fail());
    }

    {
        // Check non-parsable values
        std::istringstream is(
            "1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        swc_record record;
        is >> record;
        EXPECT_TRUE(is.fail());
    }
}

TEST(swc_parser, invalid_input_parse)
{
    using namespace nest::mc::io;

    {
        // check incomplete lines; missing parent
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830\n");
        EXPECT_THROW(parse_swc_file(is), swc_error);
    }

    {
        // Check non-parsable values
        std::istringstream is(
            "1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        EXPECT_THROW(parse_swc_file(is), swc_error);
    }

    {
        // Check invalid record type
        std::istringstream is(
            "1 10 14.566132 34.873772 7.857000 0.717830 -1\n");
        EXPECT_THROW(parse_swc_file(is), swc_error);
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

        EXPECT_THROW(parse_swc_file(is), swc_error);
    }
}

TEST(swc_parser, valid_input)
{
    using namespace nest::mc::io;

    {
        // check empty file; no record may be parsed
        swc_record record, record_orig;
        std::istringstream is("");
        is >> record;

        EXPECT_TRUE(is.eof());
        EXPECT_TRUE(is.fail());
        EXPECT_FALSE(is.bad());
        expect_record_equals(record_orig, record);
    }

    {
        // check comment-only file not ending with a newline;
        // no record may be parsed
        swc_record record, record_orig;
        std::istringstream is("#comment\n#comment");
        is >> record;

        EXPECT_TRUE(is.eof());
        EXPECT_TRUE(is.fail());
        EXPECT_FALSE(is.bad());
        expect_record_equals(record_orig, record);
    }

    {
        // check comment not starting at first character
        swc_record record, record_orig;
        std::istringstream is("   #comment\n");
        is >> record;

        EXPECT_TRUE(is.eof());
        EXPECT_TRUE(is.fail());
        EXPECT_FALSE(is.bad());
        expect_record_equals(record_orig, record);
    }

    {
        // check whitespace lines
        swc_record record;
        std::stringstream is;
        is << "#comment\n";
        is << "      \t\n";
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";

        is >> record;
        EXPECT_TRUE(is);

        swc_record record_expected(
            swc_record::kind::soma,
            0, 14.566132, 34.873772, 7.857000, 0.717830, -1);
        expect_record_equals(record_expected, record);
    }

    {
        // check windows eol
        swc_record record;
        std::stringstream is;
        is << "#comment\r\n";
        is << "\r\n";
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\r\n";

        is >> record;
        EXPECT_TRUE(is);

        swc_record record_expected(
            swc_record::kind::soma,
            0, 14.566132, 34.873772, 7.857000, 0.717830, -1);
        expect_record_equals(record_expected, record);
    }

    {
        // check old-style mac eol; these eol are treated as simple whitespace
        // characters, so should look line a long comment.
        swc_record record;
        std::stringstream is;
        is << "#comment\r";
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\r";

        is >> record;
        EXPECT_TRUE(is.eof());
        EXPECT_TRUE(is.fail());
        EXPECT_FALSE(is.bad());
    }

    {
        // check last line case (no newline at the end)
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830 -1");
        swc_record record;
        is >> record;
        EXPECT_TRUE(is.eof());
        EXPECT_FALSE(is.fail());
        EXPECT_FALSE(is.bad());
        EXPECT_EQ(0, record.id);    // zero-based indexing
        EXPECT_EQ(swc_record::kind::soma, record.type);
        EXPECT_FLOAT_EQ(14.566132, record.x);
        EXPECT_FLOAT_EQ(34.873772, record.y);
        EXPECT_FLOAT_EQ( 7.857000, record.z);
        EXPECT_FLOAT_EQ( 0.717830, record.r);
        EXPECT_FLOAT_EQ( -1, record.parent_id);
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

        using swc_iter = std::istream_iterator<swc_record>;
        swc_iter end;

        std::size_t nr_records = 0;
        for (swc_iter i = swc_iter(swc_input); i!=end; ++i) {
            ASSERT_LT(nr_records, records_orig.size());
            expect_record_equals(records_orig[nr_records], *i);
            ++nr_records;
        }
        EXPECT_EQ(2u, nr_records);
    }
}

TEST(swc_parser, from_allen_db)
{
    using namespace nest::mc::io;

    std::string datadir{DATADIR};
    auto fname = datadir + "/example.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // load the record records into a std::vector
    std::vector<swc_record> nodes = parse_swc_file(fid);

    // verify that the correct number of nodes was read
    EXPECT_EQ(1058u, nodes.size());
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

        EXPECT_THROW(parse_swc_file(is), swc_error);
    }

    {
        // Check multiple trees
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        EXPECT_THROW(parse_swc_file(is), swc_error);
    }

    {
        // Check unsorted input
        std::stringstream is;
        is << "3 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";

        std::array<swc_record::id_type, 4> expected_id_list = {{ 0, 1, 2, 3 }};

        auto records = parse_swc_file(is);
        ASSERT_EQ(expected_id_list.size(), records.size());

        for (unsigned i = 0; i< expected_id_list.size(); ++i) {
            EXPECT_EQ(expected_id_list[i], records[i].id);
        }
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

        auto records = parse_swc_file(is);
        ASSERT_EQ(expected_id_list.size(), records.size());
        for (unsigned i = 0; i< expected_id_list.size(); ++i) {
            EXPECT_EQ(expected_id_list[i], records[i].id);
            EXPECT_EQ(expected_parent_list[i], records[i].parent_id);
        }
    }
}

TEST(swc_parser, raw)
{
    using namespace nest::mc::io;

    {
        // Check valid usage
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        using swc_iter = std::istream_iterator<swc_record>;
        std::vector<swc_record> records{swc_iter(is), swc_iter()};

        EXPECT_EQ(4u, records.size());
        EXPECT_EQ(3, records.back().id);
    }

    {
        // Check parse error context
        std::stringstream is;
        is << "1 1 14.566132 34.873772 7.857000 0.717830 -1\n";
        is << "2 2 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "3 10 14.566132 34.873772 7.857000 0.717830 1\n";
        is << "4 2 14.566132 34.873772 7.857000 0.717830 1\n";

        try {
            parse_swc_file(is);
            ADD_FAILURE() << "expected an exception\n";
        }
        catch (const swc_error& e) {
            EXPECT_EQ(3u, e.line_number);
        }
    }

    {
        // Test empty range
        std::stringstream is("");
        using swc_iter = std::istream_iterator<swc_record>;
        std::vector<swc_record> records{swc_iter(is), swc_iter()};

        EXPECT_TRUE(records.empty());
    }
}

TEST(swc_io, cell_construction) {
    using namespace nest::mc;

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
    is << "2 3 0.1 1.2 1.2 1.3 1\n";
    is << "3 3 1.0 2.0 2.2 1.1 2\n";
    is << "4 3 1.5 3.3 1.3 2.2 3\n";
    is << "5 3 2.5 5.3 2.5 0.7 3\n";
    is << "6 3 3.5 2.3 3.7 3.4 5\n";

    using point_type = point<double>;
    std::vector<point_type> points = {
        { 0.0, 0.0, 0.0 },
        { 0.1, 1.2, 1.2 },
        { 1.0, 2.0, 2.2 },
        { 1.5, 3.3, 1.3 },
        { 2.5, 5.3, 2.5 },
        { 3.5, 2.3, 3.7 },
    };

    // swc -> morphology
    auto morph = io::swc_as_morphology(io::parse_swc_file(is));

    cell cell = make_cell(morph, true);
    EXPECT_TRUE(cell.has_soma());
    EXPECT_EQ(4u, cell.num_segments());

    EXPECT_EQ(norm(points[1]-points[2]), cell.cable(1)->length());
    EXPECT_EQ(norm(points[2]-points[3]), cell.cable(2)->length());
    EXPECT_EQ(norm(points[2]-points[4]) + norm(points[4]-points[5]),
              cell.cable(3)->length());


    // Check each segment separately
    EXPECT_TRUE(cell.segment(0)->is_soma());
    EXPECT_EQ(2.1, cell.soma()->radius());
    EXPECT_EQ(point_type(0, 0, 0), cell.soma()->center());

    for (auto i = 1u; i < cell.num_segments(); ++i) {
        EXPECT_TRUE(cell.segment(i)->is_dendrite());
    }

    EXPECT_EQ(1u, cell.cable(1)->num_sub_segments());
    EXPECT_EQ(1u, cell.cable(2)->num_sub_segments());
    EXPECT_EQ(2u, cell.cable(3)->num_sub_segments());

    // We asked to use the same discretization as in the SWC, so check number of compartments too.
    EXPECT_EQ(1u, cell.cable(1)->num_compartments());
    EXPECT_EQ(1u, cell.cable(2)->num_compartments());
    EXPECT_EQ(2u, cell.cable(3)->num_compartments());

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

// check that simple ball and stick model with one dendrite attached to a soma
// which is used in the validation tests can be loaded from file and matches
// the one generated with the C++ interface
TEST(swc_parser, from_file_ball_and_stick)
{
    std::string datadir{DATADIR};
    auto fname = datadir + "/ball_and_stick.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // read the file into a cell object
    auto bas_cell = make_cell(io::swc_as_morphology(io::parse_swc_file(fid)));

    // verify that the correct number of nodes was read
    EXPECT_EQ(2u, bas_cell.num_segments());
    EXPECT_EQ(2u, bas_cell.num_compartments());

    // make an equivalent cell via C++ interface
    cell local_cell;
    local_cell.add_soma(6.30785);
    local_cell.add_cable(0, section_kind::dendrite, 0.5, 0.5, 200);

    EXPECT_TRUE(cell_basic_equality(local_cell, bas_cell));
}
