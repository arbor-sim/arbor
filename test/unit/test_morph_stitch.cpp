#include <unordered_map>
#include <string>
#include <vector>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/stitch.hpp>

#include <arborio/label_parse.hpp>

#include <gtest/gtest.h>
#include "morph_pred.hpp"

using namespace arb;
using namespace arborio::literals;
using testing::region_eq;

TEST(morph, stitch_none_or_one) {
    stitch_builder B;

    stitched_morphology sm0(B);
    EXPECT_TRUE(sm0.morphology().empty());

    mpoint p1{1, 2, 3, 0.5}, p2{2, 4, 5, 1.};
    B.add({"first", p1, p2, 3});

    stitched_morphology sm1(std::move(B));
    morphology m1 = sm1.morphology();

    msegment seg0 = m1.branch_segments(0).front();
    EXPECT_EQ(3, seg0.tag);
    EXPECT_EQ(p1, seg0.prox);
    EXPECT_EQ(p2, seg0.dist);

    mprovider p(m1, sm1.labels("stitch:"));
    EXPECT_TRUE(region_eq(p, "stitch:first"_lab, reg::segment(0)));
}

TEST(morph, stitch_two) {
    {
        // p1 ===== p2 ===== p3

        mpoint p1{1, 2, 3, 0.5}, p2{2, 4, 5, 1.}, p3{3, 6, 7, 1.5};
        stitch_builder B;

        B.add({"0", p1, p2, 0})
         .add({"1", p3, 1}, "0");

        stitched_morphology sm(std::move(B));
        morphology m = sm.morphology();

        ASSERT_EQ(1u, m.num_branches());
        ASSERT_EQ(2u, m.branch_segments(0).size());

        msegment seg0 = m.branch_segments(0)[0];
        msegment seg1 = m.branch_segments(0)[1];

        EXPECT_EQ(p1, seg0.prox);
        EXPECT_EQ(p2, seg0.dist);
        EXPECT_EQ(p2, seg1.prox);
        EXPECT_EQ(p3, seg1.dist);

        mprovider p(m, sm.labels("stitch:"));
        EXPECT_TRUE(region_eq(p, "stitch:0"_lab, reg::segment(0)));
        EXPECT_TRUE(region_eq(p, "stitch:1"_lab, reg::segment(1)));
    }
    {
        // p1 ===== p2
        //  \.
        //   \.
        //    p3

        mpoint p1{1, 2, 3, 0.5}, p2{2, 4, 5, 1.}, p3{3, 6, 7, 1.5};
        stitch_builder B;

        B.add({"0", p1, p2, 0})
         .add({"1", p3, 1}, "0", 0);

        stitched_morphology sm(std::move(B));
        morphology m = sm.morphology();

        ASSERT_EQ(2u, m.num_branches());
        ASSERT_EQ(1u, m.branch_segments(0).size());
        ASSERT_EQ(1u, m.branch_segments(1).size());

        EXPECT_EQ(mnpos, m.branch_parent(0));
        EXPECT_EQ(mnpos, m.branch_parent(1));

        msegment seg0 = m.branch_segments(0)[0];
        msegment seg1 = m.branch_segments(1)[0];

        EXPECT_EQ(p1, seg0.prox);
        EXPECT_EQ(p1, seg1.prox);

        mprovider p(m, sm.labels("stitch:"));
        // Branch ordering is arbitrary, so check both possibilities:
        if (seg0.dist == p2) {
            EXPECT_TRUE(region_eq(p, "stitch:0"_lab, reg::segment(0)));
            EXPECT_TRUE(region_eq(p, "stitch:1"_lab, reg::segment(1)));
        }
        else {
            ASSERT_EQ(p2, seg1.dist);
            EXPECT_TRUE(region_eq(p, "stitch:0"_lab, reg::segment(1)));
            EXPECT_TRUE(region_eq(p, "stitch:1"_lab, reg::segment(0)));
        }
    }
    {
        // p1 ==x== p2
        //      \.
        //       \.
        //        p3

        mpoint p1{1, 2, 3, 0.5}, p2{2, 4, 5, 1.}, p3{3, 6, 7, 1.5};
        stitch_builder B;

        B.add({"0", p1, p2, 0})
         .add({"1", p3, 1}, "0", 0.5);

        stitched_morphology sm(std::move(B));
        morphology m = sm.morphology();

        ASSERT_EQ(3u, m.num_branches());
        ASSERT_EQ(1u, m.branch_segments(0).size());
        ASSERT_EQ(1u, m.branch_segments(1).size());
        ASSERT_EQ(1u, m.branch_segments(2).size());

        msegment seg0 = m.branch_segments(0)[0];
        msegment seg1 = m.branch_segments(1)[0];
        msegment seg2 = m.branch_segments(2)[0];

        EXPECT_EQ(p1, seg0.prox);
        mpoint x = lerp(p1, p2, 0.5);
        EXPECT_EQ(x, seg0.dist);

        EXPECT_EQ(x, seg1.prox);
        EXPECT_EQ(x, seg2.prox);

        mprovider p(m, sm.labels("stitch:"));
        // Branch ordering is arbitrary, so check both possibilities:
        if (seg2.dist == p2) {
            EXPECT_TRUE(region_eq(p, "stitch:0"_lab, join(reg::segment(0), reg::segment(2))));
            EXPECT_TRUE(region_eq(p, "stitch:1"_lab, reg::segment(1)));
        }
        else {
            ASSERT_EQ(p2, seg1.dist);
            EXPECT_TRUE(region_eq(p, "stitch:0"_lab, join(reg::segment(0), reg::segment(1))));
            EXPECT_TRUE(region_eq(p, "stitch:1"_lab, reg::segment(2)));
            EXPECT_TRUE(region_eq(p, "(segment 2)"_reg, reg::segment(2)));
            EXPECT_TRUE(region_eq(p, "(region \"stitch:1\")"_reg, reg::segment(2)));
        }
    }
}

TEST(morph, stitch_errors) {
    mpoint p1{1, 2, 3, 0.5}, p2{2, 4, 5, 1.}, p3{3, 6, 7, 1.5};
    stitch_builder B;

    B.add({"0", p1, p2, 0});
    ASSERT_THROW(B.add({"0", p3, 0}, "0", 0.5), duplicate_stitch_id);
    ASSERT_THROW(B.add({"1", p3, 0}, "x", 0.5), no_such_stitch);
    ASSERT_THROW(B.add({"1", p3, 0}, "0", 1.5), invalid_stitch_position);

    stitch_builder C;
    ASSERT_THROW(C.add({"0", p1, 0}), missing_stitch_start);
}

TEST(morph, stitch_complex) {
    //                p[8]
    //                  |
    // p[0]---x----x----x---p[1]---x---p[2]
    //        |    |    |          |
    //      p[3] p[4] p[5]--p[7]   p[6]

    mpoint p[] = {
        {0, 1, 0, 1.},
        {4, 1, 0, 1.},
        {6, 1, 0, 1.},
        {1, 0, 0, 1.},
        {2, 0, 0, 1.},
        {3, 0, 0, 1.},
        {5, 0, 0, 1.},
        {4, 0, 0, 1.},
        {3, 2, 0, 1.}
    };

    stitch_builder B;

    // (labels chosen to reflect distal point)
    B.add({"1", p[0], p[1]})
     .add({"2", p[2]}, "1")
     .add({"3", p[3]}, "1", 0.25)
     .add({"4", p[4]}, "1", 0.50)
     .add({"5", p[5]}, "1", 0.75)
     .add({"6", p[6]}, "2", 0.50)
     .add({"7", p[7]}, "5")
     .add({"8", p[8]}, "1", 0.75);

    stitched_morphology sm(std::move(B));
    morphology m = sm.morphology();
    mprovider P(m, sm.labels());

    EXPECT_EQ(10u, m.num_branches());

    EXPECT_EQ(4u, sm.segments("1").size());
    EXPECT_EQ(2u, sm.segments("2").size());
    EXPECT_EQ(1u, sm.segments("3").size());
    EXPECT_EQ(1u, sm.segments("4").size());
    EXPECT_EQ(1u, sm.segments("5").size());
    EXPECT_EQ(1u, sm.segments("6").size());
    EXPECT_EQ(1u, sm.segments("7").size());
    EXPECT_EQ(1u, sm.segments("8").size());

    auto region_prox_eq = [&P, place = place_pwlin(m)](region r, mpoint p) {
        mlocation_list ls = thingify(ls::most_proximal(r), P);
        if (ls.empty()) {
            return ::testing::AssertionFailure() << "region " << r << " is empty";
        }
        else if (ls.size()>1u) {
            return ::testing::AssertionFailure() << "region " << r << " has multiple proximal points";
        }

        mpoint prox = place.at(ls.front());
        if (prox!=p) {
            return ::testing::AssertionFailure() << "region " << r << " proximal point " << prox << " is not equal to " << p;
        }

        return ::testing::AssertionSuccess();
    };

    EXPECT_TRUE(region_prox_eq(sm.stitch("2"), p[1]));
    EXPECT_TRUE(region_prox_eq(sm.stitch("3"), mpoint{1, 1, 0, 1.}));
    EXPECT_TRUE(region_prox_eq(sm.stitch("4"), mpoint{2, 1, 0, 1.}));
    EXPECT_TRUE(region_prox_eq(sm.stitch("5"), mpoint{3, 1, 0, 1.}));
    EXPECT_TRUE(region_prox_eq(sm.stitch("6"), mpoint{5, 1, 0, 1.}));
    EXPECT_TRUE(region_prox_eq(sm.stitch("7"), p[5]));
    EXPECT_TRUE(region_prox_eq(sm.stitch("8"), mpoint{3, 1, 0, 1.}));
}

