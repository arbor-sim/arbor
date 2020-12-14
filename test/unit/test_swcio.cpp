#include <array>
#include <iostream>
#include <fstream>
#include <sstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/swcio.hpp>

#include "../gtest.h"


// Path to data directory can be overriden at compile time.
#if !defined(DATADIR)
#   define DATADIR "../data"
#endif

using namespace arborio;
using arb::segment_tree;
using arb::mpoint;
using arb::mnpos;

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

TEST(swc_parser, bad_parse) {
    {
        std::string bad1 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "5 1 0.1 0.2 0.3 0.4 4\n";

        EXPECT_THROW(parse_swc(bad1), swc_no_such_parent);
    }

    {
        std::string bad2 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "4 1 0.1 0.2 0.3 0.4 -1\n";

        EXPECT_THROW(parse_swc(bad2), swc_no_such_parent);
    }

    {
        std::string bad3 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 3\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_THROW(parse_swc(bad3), swc_record_precedes_parent);
    }

    {
        std::string bad4 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_THROW(parse_swc(bad4), swc_duplicate_record_id);
    }

    {
        std::string bad5 =
            "1 1 0.1 0.2 0.3 0.4 -3\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "4 1 0.1 0.2 0.3 0.4 -1\n";

        EXPECT_THROW(parse_swc(bad5), swc_no_such_parent);
    }
}

TEST(swc_parser, valid_parse) {
    // Non-contiguous is okay.
    {
        std::string valid1 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n"
            "5 1 0.1 0.2 0.3 0.4 3\n"; // non-contiguous

        EXPECT_NO_THROW(parse_swc(valid1));
    }

    // As is out of order.
    {
        std::string valid2 =
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "3 1 0.1 0.2 0.3 0.4 2\n" // out of order
            "2 1 0.1 0.2 0.3 0.4 1\n"
            "4 1 0.1 0.2 0.3 0.4 3\n";

        EXPECT_NO_THROW(parse_swc(valid2));
    }

    //  With comments
    {
        std::string valid3 =
            "# Hello\n"
            "# world.\n";

        swc_data data = parse_swc(valid3);
        EXPECT_EQ("Hello\nworld.\n", data.metadata());
        EXPECT_TRUE(data.records().empty());
    }

    // Non-contiguous, out of order records with comments.
    {
        std::string valid4 =
            "# Some people put\n"
            "# <xml /> in here!\n"
            "1 1 0.1 0.2 0.3 0.4 -1\n"
            "2 1 0.3 0.4 0.5 0.3 1\n"
            "5 2 0.2 0.6 0.8 0.2 2\n"
            "4 0 0.2 0.8 0.6 0.3 2";

        swc_data data = parse_swc(valid4);
        EXPECT_EQ("Some people put\n<xml /> in here!\n", data.metadata());
        ASSERT_EQ(4u, data.records().size());
        EXPECT_EQ(swc_record(1, 1, 0.1, 0.2, 0.3, 0.4, -1), data.records()[0]);
        EXPECT_EQ(swc_record(2, 1, 0.3, 0.4, 0.5, 0.3, 1), data.records()[1]);
        EXPECT_EQ(swc_record(4, 0, 0.2, 0.8, 0.6, 0.3, 2), data.records()[2]);
        EXPECT_EQ(swc_record(5, 2, 0.2, 0.6, 0.8, 0.2, 2), data.records()[3]);

        // Trailing garbage is ignored in data records.
        std::string valid3 =
            "# Some people put\n"
            "# <xml /> in here!\n"
            "1 1 0.1 0.2 0.3 0.4 -1 # what is that?\n"
            "2 1 0.3 0.4 0.5 0.3 1 moooooo\n"
            "3 2 0.2 0.6 0.8 0.2 2 # it is a cow!\n"
            "4 0 0.2 0.8 0.6 0.3 2";

        swc_data data2 = parse_swc(valid4);
        EXPECT_EQ(data.records(), data2.records());
    }
}

TEST(swc_parser, stream_validity) {
    {
        std::string valid =
                "# metadata\n"
                "1 1 0.1 0.2 0.3 0.4 -1\n"
                "2 1 0.1 0.2 0.3 0.4 1\n";

        std::istringstream is(valid);

        auto data = parse_swc(is);
        ASSERT_EQ(2u, data.records().size());
        EXPECT_TRUE(data.metadata() == "metadata\n");
        EXPECT_TRUE(is.eof());
    }
    {
        std::string valid =
                "# metadata\n"
                "\n"
                "1 1 0.1 0.2 0.3 0.4 -1\n"
                "2 1 0.1 0.2 0.3 0.4 1\n";

        std::istringstream is(valid);

        auto data = parse_swc(is);
        ASSERT_EQ(0u, data.records().size());
        EXPECT_TRUE(data.metadata() == "metadata\n");
        EXPECT_TRUE(is.good());

        is >> std::ws;
        data = parse_swc(is);
        ASSERT_EQ(2u, data.records().size());
        EXPECT_TRUE(data.metadata().empty());
        EXPECT_TRUE(is.eof());
    }
    {
        std::string invalid =
                "# metadata\n"
                "1 1 0.1 0.2 0.3 \n"
                "2 1 0.1 0.2 0.3 0.4 1\n";

        std::istringstream is(invalid);

        auto data = parse_swc(is);
        ASSERT_EQ(0u, data.records().size());
        EXPECT_TRUE(data.metadata() == "metadata\n");
        EXPECT_FALSE(is.good());
    }
    {
        std::string invalid =
                "# metadata\n"
                "1 1 0.1 0.2 0.3 \n"
                "2 1 0.1 0.2 0.3 0.4 1\n";

        std::istringstream is(invalid);

        auto data = parse_swc(is);
        EXPECT_TRUE(data.metadata() == "metadata\n");
        ASSERT_EQ(0u, data.records().size());
        EXPECT_TRUE(data.metadata() == "metadata\n");
        EXPECT_FALSE(is.good());

        is >> std::ws;
        data = parse_swc(is);
        ASSERT_EQ(0u, data.records().size());
        EXPECT_TRUE(data.metadata().empty());
        EXPECT_FALSE(is.good());
    }

}

TEST(swc_parser, arbor_complaint) {
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

        auto morpho = load_swc_arbor(swc);
        ASSERT_EQ(3u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);

        EXPECT_EQ(1u, segs_0.size());
        EXPECT_EQ(2u, segs_1.size());
        EXPECT_EQ(1u, segs_2.size());
        
        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);

        EXPECT_EQ(3,  segs_1[0].tag);
        EXPECT_EQ(p1, segs_1[0].prox);
        EXPECT_EQ(p2, segs_1[0].dist);

        EXPECT_EQ(3,  segs_1[1].tag);
        EXPECT_EQ(p2, segs_1[1].prox);
        EXPECT_EQ(p4, segs_1[1].dist);

        EXPECT_EQ(2,  segs_2[0].tag);
        EXPECT_EQ(p1, segs_2[0].prox);
        EXPECT_EQ(p3, segs_2[0].dist);
    }
    {
        // Otherwise, ensure segment ends and tags correspond.
        mpoint p0{0.1, 0.2, 0.3, 0.4};
        mpoint p1{0.3, 0.4, 0.5, 0.3};
        mpoint p2{0.2, 0.8, 0.6, 0.3};
        mpoint p3{0.2, 0.6, 0.8, 0.2};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius, 1},
            {3, 2, p2.x, p2.y, p2.z, p2.radius, 1},
            {4, 3, p3.x, p3.y, p3.z, p3.radius, 2},
        };

        auto morpho = load_swc_arbor(swc);
        ASSERT_EQ(2u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(mnpos, morpho.branch_parent(1));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);

        EXPECT_EQ(2u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());

        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);

        EXPECT_EQ(3,  segs_0[1].tag);
        EXPECT_EQ(p1, segs_0[1].prox);
        EXPECT_EQ(p3, segs_0[1].dist);

        EXPECT_EQ(2,  segs_1[0].tag);
        EXPECT_EQ(p0, segs_1[0].prox);
        EXPECT_EQ(p2, segs_1[0].dist);
    }
}

TEST(swc_parser, not_arbor_complaint) {
    {
        // Missing parent record will throw.
        std::vector<swc_record> swc{
            {1, 1, 0., 0., 0., 1., -1},
            {5, 3, 1., 1., 1., 1., 2}
        };
        EXPECT_THROW(load_swc_arbor(swc), swc_no_such_parent);
    }
    {
        // A single SWC record will throw.
        std::vector<swc_record> swc{
            {1, 1, 0., 0., 0., 1., -1}
        };
        EXPECT_THROW(load_swc_arbor(swc), swc_spherical_soma);
    }
    {
        std::vector<swc_record> swc{
            {1, 4, 0.1, 0.2, 0.3, 0.4, -1},
            {2, 6, 0.1, 0.2, 0.3, 0.4,  1},
            {3, 6, 0.1, 0.2, 0.3, 0.4,  2},
            {4, 6, 0.1, 0.2, 0.3, 0.4,  1}
        };
        EXPECT_THROW(load_swc_arbor(swc), swc_spherical_soma);
    }
}

TEST(swc_parser, allen_compliant) {
    using namespace arborio;
    {
        // One-point soma; interpretted as 1 segment
        mpoint p0{0, 0, 0, 10};
        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1}
        };
        auto morpho = load_swc_allen(swc);

        mpoint prox{p0.x, p0.y-p0.radius, p0.z, p0.radius};
        mpoint dist{p0.x, p0.y+p0.radius, p0.z, p0.radius};

        ASSERT_EQ(1u, morpho.num_branches());
        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(1u, segs_0.size());

        EXPECT_EQ(1,     segs_0[0].tag);
        EXPECT_EQ(prox,  segs_0[0].prox);
        EXPECT_EQ(dist,  segs_0[0].dist);
    }
    {
        // One-point soma, two-point dendrite
        mpoint p0{0,   0, 0, 10};
        mpoint p1{0,   0, 0,  5};
        mpoint p2{0, 200, 0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        auto morpho = load_swc_allen(swc);

        mpoint prox{0, -10, 0, 10};
        mpoint dist{0,  10, 0, 10};

        ASSERT_EQ(1u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(2u, segs_0.size());

        EXPECT_EQ(1,    segs_0[0].tag);
        EXPECT_EQ(prox, segs_0[0].prox);
        EXPECT_EQ(dist, segs_0[0].dist);

        EXPECT_EQ(3,    segs_0[1].tag);
        EXPECT_EQ(p1,   segs_0[1].prox);
        EXPECT_EQ(p2,   segs_0[1].dist);
    }
    {
        // 1-point soma, 2-point dendrite, 2-point axon
        mpoint p0{0, 0,  0,  1};
        mpoint p1{0, 0, 10, 10};
        mpoint p2{0, 0, 20, 10};
        mpoint p3{0, 0, 21, 10};
        mpoint p4{0, 0, 30, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 2, p3.x, p3.y, p3.z, p3.radius,  1},
            {5, 2, p4.x, p4.y, p4.z, p4.radius,  4}
        };
        auto morpho = load_swc_allen(swc);

        mpoint prox{0, -1, 0, 1};
        mpoint dist{0,  1, 0, 1};

        ASSERT_EQ(2u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(mnpos, morpho.branch_parent(1));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);

        EXPECT_EQ(2u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());

        EXPECT_EQ(1,     segs_0[0].tag);
        EXPECT_EQ(prox,  segs_0[0].prox);
        EXPECT_EQ(dist,  segs_0[0].dist);

        EXPECT_EQ(3,   segs_0[1].tag);
        EXPECT_EQ(p1,  segs_0[1].prox);
        EXPECT_EQ(p2,  segs_0[1].dist);

        EXPECT_EQ(2,   segs_1[0].tag);
        EXPECT_EQ(p3,  segs_1[0].prox);
        EXPECT_EQ(p4,  segs_1[0].dist);
    }
}

TEST(swc_parser, not_allen_compliant) {
    using namespace arborio;
    {
        // multi-point soma
        mpoint p0{0, 0, -10, 10};
        mpoint p1{0, 0,   0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_non_spherical_soma);
    }
    {
        // unsupported tag
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0, 200,  20, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 5, p1.x, p1.y, p1.z, p1.radius,  1}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_unsupported_tag);
    }
    {
        // 1-point soma; 2-point dendrite; 1-point axon connected to the proximal end of the dendrite
        mpoint p0{0, 0, -15, 10};
        mpoint p1{0, 0,   0, 10};
        mpoint p2{0, 0,  80, 10};
        mpoint p3{0, 0, -80, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 2, p3.x, p3.y, p3.z, p3.radius,  2}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_mismatched_tags);
    }
    {
        // 1-point soma and 1-point dendrite
        mpoint p0{0,   0, 0, 10};
        mpoint p1{0, 200, 0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_single_sample_segment);
    }
    {
        // 2-point dendrite and 1-point soma at the end
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0, 200,  20, 10};

        std::vector<swc_record> swc{
            {1, 3, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_no_soma);
    }
    {
        // non-existent parent sample
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0, 200,  20, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  4}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_record_precedes_parent);
    }
    {
        // parent sample is self
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0, 200,  20, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  2}
        };
        EXPECT_THROW(load_swc_allen(swc), swc_record_precedes_parent);
    }
}

TEST(swc_parser, neuron_compliant) {
    using namespace arborio;
    {
        // One-point soma; interpretted as 1 segment

        mpoint p0{0, 0, 0, 10};
        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint prox{p0.x, p0.y-p0.radius, p0.z, p0.radius};
        mpoint dist{p0.x, p0.y+p0.radius, p0.z, p0.radius};

        ASSERT_EQ(1u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(1u, segs_0.size());

        EXPECT_EQ(1,     segs_0[0].tag);
        EXPECT_EQ(prox,  segs_0[0].prox);
        EXPECT_EQ(dist,  segs_0[0].dist);
    }
    {
        // Two-point soma; interpretted as 1 segment
        mpoint p0{0, 0, -10, 10};
        mpoint p1{0, 0,   0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1}
        };
        auto morpho = load_swc_neuron(swc);

        ASSERT_EQ(1u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(1u, segs_0.size());

        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);
    }
    {
        // Three-point soma; interpretted as 2 segments
        mpoint p0{0, 0, -10, 10};
        mpoint p1{0, 0,   0, 10};
        mpoint p2{0, 0,  10, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        auto morpho = load_swc_neuron(swc);

        ASSERT_EQ(1u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(2u, segs_0.size());

        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);

        EXPECT_EQ(1,  segs_0[1].tag);
        EXPECT_EQ(p1, segs_0[1].prox);
        EXPECT_EQ(p2, segs_0[1].dist);
    }
    {
        // 6-point soma; interpretted as 5 segments
        mpoint p0{0, 0,  -5, 2};
        mpoint p1{0, 0,   0, 5};
        mpoint p2{0, 0,   2, 6};
        mpoint p3{0, 0,   6, 1};
        mpoint p4{0, 0,  10, 7};
        mpoint p5{0, 0,  15, 2};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 1, p3.x, p3.y, p3.z, p3.radius,  3},
            {5, 1, p4.x, p4.y, p4.z, p4.radius,  4},
            {6, 1, p5.x, p5.y, p5.z, p5.radius,  5}
        };
        auto morpho = load_swc_neuron(swc);

        ASSERT_EQ(1u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));

        auto segs_0 = morpho.branch_segments(0);

        EXPECT_EQ(5u, segs_0.size());

        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);

        EXPECT_EQ(1,  segs_0[1].tag);
        EXPECT_EQ(p1, segs_0[1].prox);
        EXPECT_EQ(p2, segs_0[1].dist);

        EXPECT_EQ(1,  segs_0[2].tag);
        EXPECT_EQ(p2, segs_0[2].prox);
        EXPECT_EQ(p3, segs_0[2].dist);

        EXPECT_EQ(1,  segs_0[3].tag);
        EXPECT_EQ(p3, segs_0[3].prox);
        EXPECT_EQ(p4, segs_0[3].dist);

        EXPECT_EQ(1,  segs_0[4].tag);
        EXPECT_EQ(p4, segs_0[4].prox);
        EXPECT_EQ(p5, segs_0[4].dist);
    }
    {
        // One-point soma, two-point dendrite
        mpoint p0{0,   0, 0, 10};
        mpoint p1{0,   0, 0,  5};
        mpoint p2{0, 200, 0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint prox{0, -10, 0, 10};
        mpoint dist{0,  10, 0, 10};

        ASSERT_EQ(3u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);

        EXPECT_EQ(1u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());
        EXPECT_EQ(1u, segs_2.size());

        EXPECT_EQ(1,    segs_0[0].tag);
        EXPECT_EQ(prox, segs_0[0].prox);
        EXPECT_EQ(p0,   segs_0[0].dist);

        EXPECT_EQ(1,    segs_1[0].tag);
        EXPECT_EQ(p0,   segs_1[0].prox);
        EXPECT_EQ(dist, segs_1[0].dist);

        EXPECT_EQ(3,  segs_2[0].tag);
        EXPECT_EQ(p1, segs_2[0].prox);
        EXPECT_EQ(p2, segs_2[0].dist);
    }
    {
        // 6-point soma, 2-point dendrite
        mpoint p0{0, 0,  -5, 2};
        mpoint p1{0, 0,   0, 5};
        mpoint p2{0, 0,   2, 6};
        mpoint p3{0, 0,   6, 1};
        mpoint p4{0, 0,  10, 7};
        mpoint p5{0, 0,  15, 2};
        mpoint p6{0, 0,  16, 1};
        mpoint p7{0, 0,  55, 9};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 1, p3.x, p3.y, p3.z, p3.radius,  3},
            {5, 1, p4.x, p4.y, p4.z, p4.radius,  4},
            {6, 1, p5.x, p5.y, p5.z, p5.radius,  5},
            {7, 3, p6.x, p6.y, p6.z, p6.radius,  6},
            {8, 3, p7.x, p7.y, p7.z, p7.radius,  7}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint mid {0, 0, 5, 2.25};

        ASSERT_EQ(3u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);

        EXPECT_EQ(3u, segs_0.size());
        EXPECT_EQ(3u, segs_1.size());
        EXPECT_EQ(1u, segs_2.size());

        EXPECT_EQ(1,  segs_0[0].tag);
        EXPECT_EQ(p0, segs_0[0].prox);
        EXPECT_EQ(p1, segs_0[0].dist);

        EXPECT_EQ(1,  segs_0[1].tag);
        EXPECT_EQ(p1, segs_0[1].prox);
        EXPECT_EQ(p2, segs_0[1].dist);

        EXPECT_EQ(1,  segs_0[2].tag);
        EXPECT_EQ(p2, segs_0[2].prox);
        EXPECT_EQ(mid,segs_0[2].dist);

        EXPECT_EQ(1,  segs_1[0].tag);
        EXPECT_EQ(mid,segs_1[0].prox);
        EXPECT_EQ(p3, segs_1[0].dist);

        EXPECT_EQ(1,  segs_1[1].tag);
        EXPECT_EQ(p3, segs_1[1].prox);
        EXPECT_EQ(p4, segs_1[1].dist);

        EXPECT_EQ(1,  segs_1[2].tag);
        EXPECT_EQ(p4, segs_1[2].prox);
        EXPECT_EQ(p5, segs_1[2].dist);

        EXPECT_EQ(3,  segs_2[0].tag);
        EXPECT_EQ(p6, segs_2[0].prox);
        EXPECT_EQ(p7, segs_2[0].dist);
    }
    {
        // Two-point soma, two-point dendrite
        mpoint p0{0,   0, -20, 10};
        mpoint p1{0,   0,   0,  4};
        mpoint p2{0,   0,   0, 10};
        mpoint p3{0, 200,   0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 3, p3.x, p3.y, p3.z, p3.radius,  3}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint mid{0, 0, -10, 7};

        ASSERT_EQ(3u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);

        EXPECT_EQ(1u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());
        EXPECT_EQ(1u, segs_2.size());

        EXPECT_EQ(1,   segs_0[0].tag);
        EXPECT_EQ(p0,  segs_0[0].prox);
        EXPECT_EQ(mid, segs_0[0].dist);

        EXPECT_EQ(1,   segs_1[0].tag);
        EXPECT_EQ(mid, segs_1[0].prox);
        EXPECT_EQ(p1,  segs_1[0].dist);

        EXPECT_EQ(3,   segs_2[0].tag);
        EXPECT_EQ(p2,  segs_2[0].prox);
        EXPECT_EQ(p3,  segs_2[0].dist);
    }
    {
        // 2-point soma; 2-point dendrite; 1-point axon connected to the proximal end of the dendrite
        mpoint p0{0, 0, -15, 10};
        mpoint p1{0, 0,   0,  3};
        mpoint p2{0, 0,   0, 10};
        mpoint p3{0, 0,  80, 10};
        mpoint p4{0, 0, -80, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 3, p3.x, p3.y, p3.z, p3.radius,  3},
            {5, 2, p4.x, p4.y, p4.z, p4.radius,  3}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint mid{0, 0, -7.5, 6.5};

        ASSERT_EQ(4u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));
        EXPECT_EQ(0u,    morpho.branch_parent(3));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);
        auto segs_3 = morpho.branch_segments(3);

        EXPECT_EQ(1u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());
        EXPECT_EQ(1u, segs_2.size());
        EXPECT_EQ(1u, segs_3.size());

        EXPECT_EQ(1,   segs_0[0].tag);
        EXPECT_EQ(p0,  segs_0[0].prox);
        EXPECT_EQ(mid, segs_0[0].dist);

        EXPECT_EQ(1,   segs_1[0].tag);
        EXPECT_EQ(mid, segs_1[0].prox);
        EXPECT_EQ(p1,  segs_1[0].dist);

        EXPECT_EQ(3,   segs_2[0].tag);
        EXPECT_EQ(p2,  segs_2[0].prox);
        EXPECT_EQ(p3,  segs_2[0].dist);

        EXPECT_EQ(2,   segs_3[0].tag);
        EXPECT_EQ(p2,  segs_3[0].prox);
        EXPECT_EQ(p4,  segs_3[0].dist);
    }
    {
        // 2-point soma, 2-point dendrite, 2-point axon
        mpoint p0{0, 0,  0,  1};
        mpoint p1{0, 0,  9,  2};
        mpoint p2{0, 0, 10, 10};
        mpoint p3{0, 0, 20, 10};
        mpoint p4{0, 0, 21, 10};
        mpoint p5{0, 0, 30, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 3, p3.x, p3.y, p3.z, p3.radius,  3},
            {5, 2, p4.x, p4.y, p4.z, p4.radius,  4},
            {6, 2, p5.x, p5.y, p5.z, p5.radius,  5}
        };
        auto morpho = load_swc_neuron(swc);

        mpoint mid{0, 0, 4.5, 1.5};

        ASSERT_EQ(3u, morpho.num_branches());

        EXPECT_EQ(mnpos, morpho.branch_parent(0));
        EXPECT_EQ(0u,    morpho.branch_parent(1));
        EXPECT_EQ(0u,    morpho.branch_parent(2));

        auto segs_0 = morpho.branch_segments(0);
        auto segs_1 = morpho.branch_segments(1);
        auto segs_2 = morpho.branch_segments(2);

        EXPECT_EQ(1u, segs_0.size());
        EXPECT_EQ(1u, segs_1.size());
        EXPECT_EQ(3u, segs_2.size());

        EXPECT_EQ(1,   segs_0[0].tag);
        EXPECT_EQ(p0,  segs_0[0].prox);
        EXPECT_EQ(mid, segs_0[0].dist);

        EXPECT_EQ(1,   segs_1[0].tag);
        EXPECT_EQ(mid, segs_1[0].prox);
        EXPECT_EQ(p1,  segs_1[0].dist);

        EXPECT_EQ(3,   segs_2[0].tag);
        EXPECT_EQ(p2,  segs_2[0].prox);
        EXPECT_EQ(p3,  segs_2[0].dist);

        EXPECT_EQ(2,   segs_2[1].tag);
        EXPECT_EQ(p3,  segs_2[1].prox);
        EXPECT_EQ(p4,  segs_2[1].dist);

        EXPECT_EQ(2,   segs_2[2].tag);
        EXPECT_EQ(p4,  segs_2[2].prox);
        EXPECT_EQ(p5,  segs_2[2].dist);
    }
}

TEST(swc_parser, not_neuron_compliant) {
    using namespace arborio;
    {
        // Two-point collocated soma
        mpoint p0{0, 0, 0, 5};
        mpoint p1{0, 0, 0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1}
        };

        EXPECT_THROW(load_swc_neuron(swc), swc_collocated_soma);
    }
    {
        // 3-point soma joined in the middle (1-0-2)
        mpoint p0{0, 0,   0, 10};
        mpoint p1{0, 0, -10, 10};
        mpoint p2{0, 0,  10, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  1}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_non_serial_soma);
    }
    {
        // 4-point branching soma
        mpoint p0{0,  0,  0, 10};
        mpoint p1{0,  0, 10, 10};
        mpoint p2{0, -5, 20, 10};
        mpoint p3{0,  5, 20, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 1, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 1, p3.x, p3.y, p3.z, p3.radius,  2}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_non_serial_soma);
    }
    {
        // 1-point soma and 1-point dendrite
        mpoint p0{0,   0, 0, 10};
        mpoint p1{0, 200, 0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 3, p1.x, p1.y, p1.z, p1.radius,  1}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_single_sample_segment);
    }
    {
        // 2-point soma and 1-point dendrite
        mpoint p0{0,   0, -10, 10};
        mpoint p1{0,   0,   0, 10};
        mpoint p2{0, 200,   0, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_single_sample_segment);
    }
    {
        // 2-point soma and two 1-point dendrite
        mpoint p0{0,  0, -20, 10};
        mpoint p1{0,  0,   0, 10};
        mpoint p2{0, -5,  80, 10};
        mpoint p3{0,  5, -90, 10};

        std::vector<swc_record> swc{
            {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
            {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
            {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
            {4, 3, p3.x, p3.y, p3.z, p3.radius,  2}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_single_sample_segment);
    }
    {
        // 2-point dendrite and 1-point soma at the end
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0, 200,  20, 10};

        std::vector<swc_record> swc{
                {1, 3, p0.x, p0.y, p0.z, p0.radius, -1},
                {2, 3, p1.x, p1.y, p1.z, p1.radius,  1},
                {3, 1, p2.x, p2.y, p2.z, p2.radius,  2}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_no_soma);
    }
    {
        // 3-point non-consecutive soma and a 2 point dendrite
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0,   0,  10, 10};
        mpoint p3{0, 200,  20, 10};
        mpoint p4{0,   0,  20,  1};

        std::vector<swc_record> swc{
                {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
                {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
                {3, 3, p2.x, p2.y, p2.z, p2.radius,  2},
                {4, 3, p3.x, p3.y, p3.z, p3.radius,  3},
                {5, 1, p4.x, p4.y, p4.z, p4.radius,  2}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_non_consecutive_soma);
    }
    {
        // 3-point soma and a 2 point dendrite connected in the middle of the soma
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0,   0,  20,  1};
        mpoint p3{0,   0,  10, 10};
        mpoint p4{0, 200,  20, 10};

        std::vector<swc_record> swc{
                {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
                {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
                {3, 1, p2.x, p2.y, p2.z, p2.radius,  2},
                {4, 3, p3.x, p3.y, p3.z, p3.radius,  2},
                {5, 3, p4.x, p4.y, p4.z, p4.radius,  4}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_branchy_soma);
    }
    {
        // non-existent parent sample
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0, 200,  20, 10};

        std::vector<swc_record> swc{
                {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
                {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
                {3, 3, p2.x, p2.y, p2.z, p2.radius,  4}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_record_precedes_parent);
    }
    {
        // parent sample is self
        mpoint p0{0,   0,   0,  1};
        mpoint p1{0,   0,  10,  1};
        mpoint p2{0, 200,  20, 10};

        std::vector<swc_record> swc{
                {1, 1, p0.x, p0.y, p0.z, p0.radius, -1},
                {2, 1, p1.x, p1.y, p1.z, p1.radius,  1},
                {3, 3, p2.x, p2.y, p2.z, p2.radius,  3}
        };
        EXPECT_THROW(load_swc_neuron(swc), swc_record_precedes_parent);
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

    auto data = parse_swc(fid);
    EXPECT_EQ(5799u, data.records().size());
}
#endif
