#include <array>
#include <iostream>
#include <fstream>
#include <sstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>
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
}

TEST(swc_parser, bad_strict) {
    {
        std::string bad6 =
            "1 7 0.1 0.2 0.3 0.4 -1\n"; // just one record

        EXPECT_THROW(parse_swc(bad6, swc_mode::relaxed), swc_spherical_soma);
    }
    {
        std::string bad3 =
            "1 4 0.1 0.2 0.3 0.4 -1\n" // solitary tag
            "2 6 0.1 0.2 0.3 0.4 1\n"
            "3 6 0.1 0.2 0.3 0.4 2\n"
            "4 6 0.1 0.2 0.3 0.4 1\n";

        EXPECT_THROW(parse_swc(bad3, swc_mode::strict), swc_spherical_soma);
        EXPECT_NO_THROW(parse_swc(bad3, swc_mode::relaxed));
    }
}

TEST(swc_parser, valid_relaxed) {
    // Non-contiguous is okay.
    {
        std::string bad1 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "5 1 0.1 0.2 0.3 0.4 3\n"; // non-contiguous

        EXPECT_NO_THROW(parse_swc(bad1, swc_mode::relaxed));
    }

    // As is out of order.
    {
        std::string bad2 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n" // out of order
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_NO_THROW(parse_swc(bad2, swc_mode::relaxed));
    }

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
        // Non-contiguous, out of order records are fine.
        std::string valid2 =
            "# Some people put\n"
            "# <xml /> in here!\n"
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.3 0.4 0.5 0.3 1\n"
            "5 2 0.2 0.6 0.8 0.2 2\n"
            "4 0 0.2 0.8 0.6 0.3 2";

        swc_data data = parse_swc(valid2, swc_mode::strict);
        EXPECT_EQ("Some people put\n<xml /> in here!\n", data.metadata);
        ASSERT_EQ(4u, data.records.size());
        EXPECT_EQ(swc_record(1, 1, 0.1, 0.2, 0.3, 0.4, -1), data.records[0]);
        EXPECT_EQ(swc_record(2, 1, 0.3, 0.4, 0.5, 0.3, 1), data.records[1]);
        EXPECT_EQ(swc_record(4, 0, 0.2, 0.8, 0.6, 0.3, 2), data.records[2]);
        EXPECT_EQ(swc_record(5, 2, 0.2, 0.6, 0.8, 0.2, 2), data.records[3]);

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

TEST(swc_parser, segment_tree) {
    {
        // Missing parent record will throw.
        std::vector<swc_record> swc{
            {1, 1, 0., 0., 0., 1., -1},
            {5, 3, 1., 1., 1., 1., 2}
        };
        EXPECT_THROW(as_segment_tree(swc), bad_swc_data);
    }
    {
        // A single SWC record will throw.
        std::vector<swc_record> swc{
            {1, 1, 0., 0., 0., 1., -1}
        };
        EXPECT_THROW(as_segment_tree(swc), bad_swc_data);
    }
    {
        // Otherwise, ensure segment ends and tags correspond.
        mpoint p0{0.1, 0.2, 0.3, 0.4};
        mpoint p1{0.3, 0.4, 0.5, 0.3};
        mpoint p2{0.2, 0.8, 0.6, 0.3};
        mpoint p3{0.2, 0.6, 0.8, 0.2};
        mpoint p4{0.4, 0.5, 0.5, 0.1};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius, 1},
            {4, 3, p2.x, p2.y, p2.z, p2.radius, 2},
            {5, 2, p3.x, p3.y, p3.z, p3.radius, 2},
            {7, 3, p4.x, p4.y, p4.z, p4.radius, 4}
        };

        segment_tree tree = as_segment_tree(swc);
        ASSERT_EQ(4u, tree.segments().size());

        EXPECT_EQ(mnpos, tree.parents()[0]);
        EXPECT_EQ(1,  tree.segments()[0].tag);
        EXPECT_EQ(p0, tree.segments()[0].prox);
        EXPECT_EQ(p1, tree.segments()[0].dist);

        EXPECT_EQ(0u, tree.parents()[1]);
        EXPECT_EQ(3,  tree.segments()[1].tag);
        EXPECT_EQ(p1, tree.segments()[1].prox);
        EXPECT_EQ(p2, tree.segments()[1].dist);

        EXPECT_EQ(0u, tree.parents()[2]);
        EXPECT_EQ(2,  tree.segments()[2].tag);
        EXPECT_EQ(p1, tree.segments()[2].prox);
        EXPECT_EQ(p3, tree.segments()[2].dist);

        EXPECT_EQ(1u, tree.parents()[3]);
        EXPECT_EQ(3,  tree.segments()[3].tag);
        EXPECT_EQ(p2, tree.segments()[3].prox);
        EXPECT_EQ(p4, tree.segments()[3].dist);
    }
}

// hipcc bug in reading DATADIR
#ifndef ARB_HIP
TEST(swc_parser, from_neuromorpho)
{
    std::string datadir{DATADIR};
    auto fname = datadir + "/pyramidal.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    auto data = parse_swc(fid, swc_mode::strict);
    EXPECT_EQ(5799u, data.records.size());
}
#endif
