#include <array>
#include <iostream>
#include <fstream>
#include <sstream>

#include <arbor/cable_cell.hpp>
#include <arbor/swcio.hpp>

#include "../gtest.h"


// Path to data directory can be overriden at compile time.
#if !defined(DATADIR)
#   define DATADIR "../data"
#endif

using namespace arb;

TEST(swc_record, construction) {
    swc_record record(1, 7, 1., 2., 3., 4., -1);
    EXPECT_EQ(record.id, 1);
    EXPECT_EQ(record.tag, 7);
    EXPECT_EQ(record.x, 1.);
    EXPECT_EQ(record.y, 2.);
    EXPECT_EQ(record.z, 3.);
    EXPECT_EQ(record.r, 4.);
    EXPECT_EQ(record.parent_id, -1);

    // Check copy ctor and copy assign.
    swc_record r2(record);
    EXPECT_EQ(record, r2);

    swc_record r3;
    EXPECT_NE(record, r3);

    r3 = record;
    EXPECT_EQ(record, r3);
}

TEST(swc_record, invalid_input) {
    swc_record dummy{2, 3, 4., 5., 6., 7., 1};
    {
        // Incomplete line: missing parent id.
        swc_record record(dummy);
        std::istringstream is("1 1 14.566132 34.873772 7.857000 0.717830\n");
        is >> record;

        EXPECT_TRUE(is.fail());
        EXPECT_EQ(dummy, record);
    }

    {
        // Bad id value.
        swc_record record(dummy);
        std::istringstream is("1a 1 14.566132 34.873772 7.857000 0.717830 -1\n");
        is >> record;

        EXPECT_TRUE(is.fail());
        EXPECT_EQ(dummy, record);
    }
}

TEST(swc_parser, bad_relaxed) {
    {
        std::string bad1 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "5 1 0.1 0.2 0.3 0.4 4\n";

        EXPECT_THROW(parse_swc(bad1, swc_mode::relaxed), swc_no_such_parent);
    }

    {
        std::string bad2 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "4 1 0.1 0.2 0.3 0.4 -1\n";

        EXPECT_THROW(parse_swc(bad2, swc_mode::relaxed), swc_no_such_parent);
    }

    {
        std::string bad3 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 3\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_THROW(parse_swc(bad3, swc_mode::relaxed), swc_record_precedes_parent);
    }

    {
        std::string bad4 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_THROW(parse_swc(bad4, swc_mode::relaxed), swc_duplicate_record_id);
    }

    {
        std::string bad5 =
            "1 1 0.1 0.2 0.3 0.4 -3\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "4 1 0.1 0.2 0.3 0.4 -1\n";

        EXPECT_THROW(parse_swc(bad5, swc_mode::relaxed), swc_no_such_parent);
    }

    {
        std::string bad6 =
            "1 1 0.1 0.2 0.3 0.4 -1\n";

        EXPECT_THROW(parse_swc(bad6, swc_mode::relaxed), swc_spherical_soma);
    }
}

TEST(swc_parser, bad_strict) {
    {
        std::string bad1 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "5 1 0.1 0.2 0.3 0.4 3\n"; // non-contiguous

        EXPECT_THROW(parse_swc(bad1, swc_mode::strict), swc_irregular_id);
        EXPECT_NO_THROW(parse_swc(bad1, swc_mode::relaxed));
    }

    {
        std::string bad2 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n" // out of order
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_THROW(parse_swc(bad2, swc_mode::strict), swc_irregular_id);
        EXPECT_NO_THROW(parse_swc(bad2, swc_mode::relaxed));
    }

    {
        std::string bad3 =
            "1 1 0.1 0.2 0.3 0.4 -1\n" // solitary tag
            "2 0 0.1 0.2 0.3 0.4 1\n"
            "3 0 0.1 0.2 0.3 0.4 2\n"
            "4 0 0.1 0.2 0.3 0.4 1\n";

        EXPECT_THROW(parse_swc(bad3, swc_mode::strict), swc_spherical_soma);
        EXPECT_NO_THROW(parse_swc(bad3, swc_mode::relaxed));
    }
}

TEST(swc_parser, valid_relaxed) {
}

TEST(swc_parser, valid_strict) {
    {
        std::string valid1 =
            "# Hello\n"
            "# world.\n";

        swc_data data = parse_swc(valid1, swc_mode::strict);
        EXPECT_EQ("Hello\nworld.\n", data.metadata);
        EXPECT_TRUE(data.records.empty());
    }

    {
        std::string valid2 =
            "# Some people put\n"
            "# <xml /> in here!\n"
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.3 0.4 0.5 0.3 1\n"
            "3 2 0.2 0.6 0.8 0.2 2\n"
            "4 0 0.2 0.8 0.6 0.3 2";

        swc_data data = parse_swc(valid2, swc_mode::strict);
        EXPECT_EQ("Some people put\n<xml /> in here!\n", data.metadata);
        ASSERT_EQ(4u, data.records.size());
        EXPECT_EQ(swc_record(1, 1, 0.1, 0.2, 0.3, 0.4, -1), data.records[0]);
        EXPECT_EQ(swc_record(2, 1, 0.3, 0.4, 0.5, 0.3, 1), data.records[1]);
        EXPECT_EQ(swc_record(3, 2, 0.2, 0.6, 0.8, 0.2, 2), data.records[2]);
        EXPECT_EQ(swc_record(4, 0, 0.2, 0.8, 0.6, 0.3, 2), data.records[3]);

        // Trailing garbage is ignored in data records.
        std::string valid3 =
            "# Some people put\n"
            "# <xml /> in here!\n"
            "1 1 0.1 0.2 0.3 0.4 -1 # what is that?\n"
            "2 1 0.3 0.4 0.5 0.3 1 moooooo\n"
            "3 2 0.2 0.6 0.8 0.2 2 # it is a cow!\n"
            "4 0 0.2 0.8 0.6 0.3 2";

        swc_data data2 = parse_swc(valid2, swc_mode::strict);
        EXPECT_EQ(data.records, data2.records);
    }
}

// hipcc bug in reading DATADIR
#ifndef ARB_HIP
TEST(swc_parser, from_allen_db)
{
    std::string datadir{DATADIR};
    auto fname = datadir + "/example.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    auto data = parse_swc(fid, swc_mode::strict);
    EXPECT_EQ(1060u, data.records.size());
}
#endif
