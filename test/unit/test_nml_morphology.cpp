#include <optional>
#include <string>
#include <vector>

#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>

#include <arborio/neuroml.hpp>

#include <gtest/gtest.h>
#include "morph_pred.hpp"

using testing::region_eq;

// Tests for basic morphology scanning and collection from XML.

TEST(neuroml, morph_badxml) {
    std::string illformed = "<wha?";

    EXPECT_THROW(arborio::neuroml{illformed}, std::runtime_error);
}

TEST(neuroml, morph_none) {
    // No NeuroML doc, with and without declaration:
    {
        std::string empty1 = R"~(<?xml version="1.0" encoding="UTF-8"?><foo/>)~";

        arborio::neuroml N1(empty1);
        EXPECT_TRUE(N1.cell_ids().empty());
        EXPECT_TRUE(N1.morphology_ids().empty());

        std::string empty2 = "<foo/>";

        arborio::neuroml N2(empty2);
        EXPECT_TRUE(N2.cell_ids().empty());
        EXPECT_TRUE(N2.morphology_ids().empty());
    }

    // Empty NeuroML doc:
    {
        std::string empty3 =
R"~(<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
</neuroml>)~";

        arborio::neuroml N3(empty3);
        EXPECT_TRUE(N3.cell_ids().empty());
        EXPECT_TRUE(N3.morphology_ids().empty());
    }
}

TEST(neuroml, morph_ids) {
    // Two top-level morphologies (m1 and m2);
    // cell c3 uses top-level morphology m1;
    // cell c4 uses internally defined morphology m4.
    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="m1"/>
<morphology id="m2"/>
<cell id="c3" morphology="m1"/>
<cell id="c4">
    <morphology id="m4"/>
</cell>
</neuroml>
)~";

    using svector = std::vector<std::string>;

    arborio::neuroml N(doc);

    svector m_ids = N.morphology_ids(); // only top-level!
    std::sort(m_ids.begin(), m_ids.end());
    EXPECT_EQ((svector{"m1", "m2"}), m_ids);

    svector c_ids = N.cell_ids();
    std::sort(c_ids.begin(), c_ids.end());
    EXPECT_EQ((svector{"c3", "c4"}), c_ids);

    arborio::nml_morphology_data mdata;

    mdata = N.cell_morphology("c4").value();
    EXPECT_EQ("c4", mdata.cell_id);
    EXPECT_EQ("m4", mdata.id);

    mdata = N.cell_morphology("c3").value();
    EXPECT_EQ("c3", mdata.cell_id);
    EXPECT_EQ("m1", mdata.id);

    EXPECT_THROW(N.cell_morphology("mr. bobbins").value(), std::bad_optional_access);
}

TEST(neuroml, simple_morphologies) {
    using namespace arb;

    // Points used in morphology definitions below.

    mpoint p0{1, -2, 3.5, 4};
    mpoint p1{3, -3.5, 4, 4.25};
    mpoint p2{3, -4, 4, 2.25};
    mpoint p3{4.5, -5, 5, 0.25};

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="m1">
    <!-- Just one segment between p0 and p1. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
</morphology>
<morphology id="m2">
    <!-- Two segments, implicit proximal, [p0 p1] [p1 p3]. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="0"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="m3">
    <!-- Two segments, explicit proximal (with gap)
         [p0 p1] [p2 p3]. -->
    <segment id="0" name="soma">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="0"/>
        <proximal x="3" y="-4" z="4" diameter="4.5"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="m4">
    <!-- Two segments, meeting at root point p0,
         [p0 p1] and [p0 p3]. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="0" fractionAlong="0.0"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="m5">
    <!-- Two segments, meeting at root point p0,
         [p0 p1] and [p0 p3], but in reverse order. -->
    <segment id="1">
        <parent segment="0" fractionAlong="0.0"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);

    {
        auto m1 = N.morphology("m1").value();
        label_dict labels;
        labels.import(m1.segments, "seg:");
        mprovider P(m1.morphology, labels);

        EXPECT_TRUE(region_eq(P, reg::named("seg:0"), reg::all()));

        place_pwlin G(P.morphology());
        EXPECT_EQ(p0, G.at(mlocation{0, 0}));
        EXPECT_EQ(p1, G.at(mlocation{0, 1}));
    }

    {
        arborio::nml_morphology_data m2 = N.morphology("m2").value();
        label_dict labels;
        labels.import(m2.segments, "seg:");
        mprovider P(m2.morphology, labels);

        mextent seg0_extent = thingify(reg::named("seg:0"), P);
        ASSERT_EQ(1u, seg0_extent.size());
        mcable seg0 = seg0_extent.cables()[0];

        mextent seg1_extent = thingify(reg::named("seg:1"), P);
        ASSERT_EQ(1u, seg1_extent.size());
        mcable seg1 = seg1_extent.cables()[0];

        EXPECT_EQ(0u, seg0.branch);
        EXPECT_EQ(0.0, seg0.prox_pos);

        EXPECT_EQ(0u, seg1.branch);
        EXPECT_EQ(seg0.dist_pos, seg1.prox_pos);
        EXPECT_EQ(1.0, seg1.dist_pos);

        place_pwlin G(P.morphology());
        EXPECT_EQ(p0, G.at(prox_loc(seg0)));
        EXPECT_EQ(p1, G.at(dist_loc(seg0)));
        EXPECT_EQ(p1, G.at(prox_loc(seg1)));
        EXPECT_EQ(p3, G.at(dist_loc(seg1)));
    }

    {
        arborio::nml_morphology_data m3 = N.morphology("m3").value();
        label_dict labels;
        labels.import(m3.segments, "seg:");
        mprovider P(m3.morphology, labels);

        mextent seg0_extent = thingify(reg::named("seg:0"), P);
        ASSERT_EQ(1u, seg0_extent.size());
        mcable seg0 = seg0_extent.cables()[0];

        mextent seg1_extent = thingify(reg::named("seg:1"), P);
        ASSERT_EQ(1u, seg1_extent.size());
        mcable seg1 = seg1_extent.cables()[0];

        EXPECT_EQ(0u, seg0.branch);
        EXPECT_EQ(0.0, seg0.prox_pos);

        EXPECT_EQ(0u, seg1.branch);
        EXPECT_EQ(seg0.dist_pos, seg1.prox_pos);
        EXPECT_EQ(1.0, seg1.dist_pos);

        place_pwlin G(P.morphology());
        auto seg0_segments = G.segments(seg0_extent);
        auto seg1_segments = G.segments(seg1_extent);

        ASSERT_EQ(1u, seg0_segments.size());
        EXPECT_EQ(p0, seg0_segments[0].prox);
        EXPECT_EQ(p1, seg0_segments[0].dist);

        ASSERT_EQ(1u, seg1_segments.size());
        EXPECT_EQ(p2, seg1_segments[0].prox);
        EXPECT_EQ(p3, seg1_segments[0].dist);
    }
    {
        for (const char* m_name: {"m4", "m5"}) {
            arborio::nml_morphology_data m4_or_5 = N.morphology(m_name).value();
            label_dict labels;
            labels.import(m4_or_5.segments, "seg:");
            mprovider P(m4_or_5.morphology, labels);

            mextent seg0_extent = thingify(reg::named("seg:0"), P);
            ASSERT_EQ(1u, seg0_extent.size());

            mextent seg1_extent = thingify(reg::named("seg:1"), P);
            ASSERT_EQ(1u, seg1_extent.size());

            place_pwlin G(P.morphology());
            auto seg0_segments = G.segments(seg0_extent);
            auto seg1_segments = G.segments(seg1_extent);

            ASSERT_EQ(1u, seg0_segments.size());
            EXPECT_EQ(p0, seg0_segments[0].prox);
            EXPECT_EQ(p1, seg0_segments[0].dist);

            ASSERT_EQ(1u, seg1_segments.size());
            EXPECT_EQ(p0, seg1_segments[0].prox);
            EXPECT_EQ(p3, seg1_segments[0].dist);
        }
    }

}

TEST(neuroml, spherical_segments) {
    using namespace arb;
    using namespace arborio::neuroml_options;

    // Spherical root segments can be translated as equivalent-area
    // cylinders oriented along the y-axis in the generated morphology.

    // Points used in morphology definitions below.

    mpoint p0{1, -2, 3.5, 4};
    mpoint p1{1, -2, 3.5, 3};
    mpoint p2{3, -3, 4.5, 5};

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="m1">
    <!-- Single zero-length segment at p0 -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="1" y="-2" z="3.5" diameter="8"/>
    </segment>
</morphology>
<morphology id="m2">
    <!-- Single zero-length segment defined between colocated p0 and p1;
         diameters differ, so should not be treated as spherical. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="1" y="-2" z="3.5" diameter="6"/>
    </segment>
</morphology>
<morphology id="m3">
    <!-- Two segments: first is zero-length with ends colocated at p0;
         second has distal point p2, and attaches at fractionAlong=0.2, but should inherit proximal point p0. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="1" y="-2" z="3.5" diameter="8"/>
    </segment>
    <segment id="1">
        <parent segment="0" fractionAlong="0.2"/>
        <distal x="3" y="-3" z="4.5" diameter="10"/>
    </segment>
</morphology>
<morphology id="m4">
    <!-- Two segments: first is between p1 and p2; second is from p2 to p2, but is not the root segment
         and should not be interpreted as a sphere. -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="6"/>
        <distal x="3" y="-3" z="4.5" diameter="10"/>
    </segment>
    <segment id="1">
        <parent segment="0"/>
        <proximal x="3" y="-3" z="4.5" diameter="10"/>
        <distal x="3" y="-3" z="4.5" diameter="10"/>
    </segment>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);

    {
        arborio::nml_morphology_data m1 = N.morphology("m1", allow_spherical_root).value();
        label_dict labels;
        labels.import(m1.segments, "seg:");
        mprovider P(m1.morphology, labels);

        EXPECT_TRUE(region_eq(P, reg::branch(0), reg::all()));
        EXPECT_TRUE(region_eq(P, reg::named("seg:0"), reg::all()));

        place_pwlin G(P.morphology());
        EXPECT_EQ(p0.radius, G.at(mlocation{0, 0}).radius);
        EXPECT_EQ(p0.radius, G.at(mlocation{0, 1}).radius);

        mpoint centre = G.at(mlocation{0, 0.5});
        EXPECT_EQ(p0, centre);

        // Only y-axis points should differ from centre.
        mpoint l0 = G.at(mlocation{0, 0});
        mpoint l1 = G.at(mlocation{0, 1});
        EXPECT_EQ(p0.x, l0.x);
        EXPECT_NE(p0.y, l0.y);
        EXPECT_EQ(p0.z, l0.z);
        EXPECT_EQ(p0.x, l1.x);
        EXPECT_NE(p0.y, l1.y);
        EXPECT_EQ(p0.z, l1.z);

        EXPECT_DOUBLE_EQ(2*p0.radius, P.embedding().branch_length(0));
    }
    {
        // With spherical root _not_ provided, treat it just as a simple zero-length segment.
        arborio::nml_morphology_data m1 = N.morphology("m1", none).value();
        label_dict labels;
        labels.import(m1.segments, "seg:");
        mprovider P(m1.morphology, labels);

        EXPECT_TRUE(region_eq(P, reg::branch(0), reg::all()));
        EXPECT_TRUE(region_eq(P, reg::named("seg:0"), reg::all()));

        place_pwlin G(P.morphology());
        EXPECT_EQ(p0, G.at(mlocation{0, 0}));
        EXPECT_EQ(p0, G.at(mlocation{0, 1}));
    }
    {
        arborio::nml_morphology_data m2 = N.morphology("m2", allow_spherical_root).value();
        label_dict labels;
        labels.import(m2.segments, "seg:");
        mprovider P(m2.morphology, labels);

        EXPECT_TRUE(region_eq(P, reg::branch(0), reg::all()));
        EXPECT_TRUE(region_eq(P, reg::named("seg:0"), reg::all()));

        // This one shouldn't be interpreted as a sphere.
        place_pwlin G(P.morphology());
        auto points = G.all_at(mlocation{0, 0});
        ASSERT_EQ(2u, points.size());
        EXPECT_TRUE((p0==points[0] && p1==points[1]) ||
                    (p0==points[1] && p1==points[0]));
    }
    {
        arborio::nml_morphology_data m3 = N.morphology("m3", allow_spherical_root).value();
        label_dict labels;
        labels.import(m3.segments, "seg:");
        mprovider P(m3.morphology, labels);
        place_pwlin G(P.morphology());

        // With segment 1 attached to spherical segment 0, we should have three branches.
        EXPECT_EQ(3u, P.morphology().num_branches());

        mlocation s0centre = thingify(ls::on_components(0.5, reg::named("seg:0")), P).at(0);
        EXPECT_EQ(p0, G.at(s0centre));

        // Compute locations of ends of segment 1.
        mlocation s1ploc = thingify(ls::most_proximal(reg::named("seg:1")), P).at(0);
        mlocation s1dloc = thingify(ls::most_distal(reg::named("seg:1")), P).at(0);
        mpoint s1p = G.at(s1ploc);
        mpoint s1d = G.at(s1dloc);

        EXPECT_TRUE(testing::point_almost_eq(p0, s1p));
        EXPECT_EQ(p2, s1d);
    }
    {
        arborio::nml_morphology_data m4 = N.morphology("m4", allow_spherical_root).value();
        label_dict labels;
        labels.import(m4.segments, "seg:");
        mprovider P(m4.morphology, labels);
        place_pwlin G(P.morphology());

        // Segment 1 should have colocated, equal radius endpoints, as it is not the root segment.
        mlocation s1ploc = thingify(ls::most_proximal(reg::named("seg:1")), P).at(0);
        mlocation s1dloc = thingify(ls::most_distal(reg::named("seg:1")), P).at(0);
        mpoint s1p = G.at(s1ploc);
        mpoint s1d = G.at(s1dloc);

        EXPECT_EQ(p2, s1p);
        EXPECT_EQ(p2, s1d);
    }
}

TEST(neuroml, segment_errors) {
    using namespace arb;

    // Points used in morphology definitions below.

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="no-proximal">
    <!-- No proximal point for root segment -->
    <segment id="0">
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
</morphology>
<morphology id="no-such-parent">
    <!-- Parent of segment 1 does not exist -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="2"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="cyclic-dependency">
    <!-- Segments 1, 2 3 form a cycle -->
    <segment id="0" name="soma">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="3"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
    <segment id="2">
        <parent segment="1"/>
        <distal x="5.5" y="-5" z="5" diameter="0.5"/>
    </segment>
    <segment id="3">
        <parent segment="2"/>
        <distal x="6.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="duplicate-id">
    <!-- Two segments with the same id -->
    <segment id="0">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
    <segment id="1">
        <parent segment="0" fractionAlong="0.0"/>
        <distal x="4.5" y="-5" z="5" diameter="0.5"/>
    </segment>
    <segment id="1">
        <parent segment="0" fractionAlong="0.0"/>
        <distal x="7.5" y="-5" z="5" diameter="0.5"/>
    </segment>
</morphology>
<morphology id="bad-segment-id">
    <!-- Segment id is a negative number -->
    <segment id="-1">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
</morphology>
<morphology id="another-bad-segment-id">
    <!-- Segment id is not a whole number -->
    <segment id="1.6">
        <proximal x="1" y="-2" z="3.5" diameter="8"/>
        <distal x="3" y="-3.5" z="4" diameter="8.5"/>
    </segment>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);

    EXPECT_THROW(N.morphology("no-proximal").value(), arborio::nml_bad_segment);
    EXPECT_THROW(N.morphology("no-such-parent").value(), arborio::nml_bad_segment);
    EXPECT_THROW(N.morphology("cyclic-dependency").value(), arborio::nml_cyclic_dependency);
    EXPECT_THROW(N.morphology("duplicate-id").value(), arborio::nml_bad_segment);
    EXPECT_THROW(N.morphology("bad-segment-id").value(), arborio::nml_bad_segment);
    EXPECT_THROW(N.morphology("another-bad-segment-id").value(), arborio::nml_bad_segment);
}

TEST(neuroml, simple_groups) {
    using namespace arb;

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="m1">
    <segment id="0">
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segment id="1">
        <parent segment="0"/>
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segment id="2">
        <parent segment="1"/>
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segmentGroup id="group-a">
        <member segment="0"/>
    </segmentGroup>
    <segmentGroup id="group-b">
        <member segment="2"/>
    </segmentGroup>
    <segmentGroup id="group-c">
        <member segment="2"/>
        <member segment="1"/>
    </segmentGroup>
</morphology>
<morphology id="m2">
    <segment id="0">
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segment id="1">
        <parent segment="0"/>
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segment id="2">
        <parent segment="1"/>
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segment id="3">
        <parent segment="2"/>
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segmentGroup id="group-a">
        <!-- segments 0 and 2 -->
        <member segment="0"/>
        <include segmentGroup="group-b"/>
    </segmentGroup>
    <segmentGroup id="group-b">
        <member segment="2"/>
    </segmentGroup>
    <segmentGroup id="group-c">
        <!-- segments 0, 1 and 2 -->
        <member segment="1"/>
        <include segmentGroup="group-a"/>
    </segmentGroup>
    <segmentGroup id="group-d">
        <!-- segments 0, 2 and 3 -->
        <include segmentGroup="group-e"/>
        <include segmentGroup="group-a"/>
    </segmentGroup>
    <segmentGroup id="group-e">
        <member segment="3"/>
    </segmentGroup>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);
    using reg::named;

    {
        arborio::nml_morphology_data m1 = N.morphology("m1").value();
        label_dict labels;
        labels.import(m1.segments);
        labels.import(m1.groups);
        mprovider P(m1.morphology, labels);

        EXPECT_TRUE(region_eq(P, named("group-a"), named("0")));
        EXPECT_TRUE(region_eq(P, named("group-b"), named("2")));
        EXPECT_TRUE(region_eq(P, named("group-c"), join(named("2"), named("1"))));
    }
    {
        arborio::nml_morphology_data m2 = N.morphology("m2").value();
        label_dict labels;
        labels.import(m2.segments);
        labels.import(m2.groups);
        mprovider P(m2.morphology, labels);

        EXPECT_TRUE(region_eq(P, named("group-a"), join(named("0"), named("2"))));
        EXPECT_TRUE(region_eq(P, named("group-c"), join(named("0"), named("1"), named("2"))));
        EXPECT_TRUE(region_eq(P, named("group-d"), join(named("0"), named("2"), named("3"))));
    }
}

TEST(neuroml, group_errors) {
    using namespace arb;

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="no-such-segment">
    <segment id="0">
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segmentGroup id="group">
        <member segment="1"/>
    </segmentGroup>
</morphology>
<morphology id="no-such-group">
    <segment id="0">
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segmentGroup id="group">
        <member segment="0"/>
        <include segmentGroup="other-group"/>
    </segmentGroup>
</morphology>
<morphology id="cyclic-dependency">
    <segment id="0">
        <proximal x="1" y="1" z="1" diameter="1"/>
        <distal x="2" y="2" z="2" diameter="2"/>
    </segment>
    <segmentGroup id="group">
        <include segmentGroup="other-group"/>
    </segmentGroup>
    <segmentGroup id="other-group">
        <include segmentGroup="group"/>
    </segmentGroup>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);

    EXPECT_THROW(N.morphology("no-such-segment").value(), arborio::nml_bad_segment_group);
    EXPECT_THROW(N.morphology("no-such-group").value(), arborio::nml_bad_segment_group);
    EXPECT_THROW(N.morphology("cyclic-dependency").value(), arborio::nml_cyclic_dependency);
}


TEST(neuroml, group_paths_subtrees) {
    using namespace arb;

    std::string doc =
R"~(
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2">
<morphology id="m1">
    <segment id="0">
        <proximal x="0" y="0" z="0" diameter="1"/>
        <distal x="1" y="0" z="0" diameter="2"/>
    </segment>
    <segment id="1">
        <parent segment="0" fractionAlong="0.5"/>
        <proximal x="0.5" y="0" z="0" diameter="1"/>
        <distal x="0.5" y="1" z="0" diameter="2"/>
    </segment>
    <segment id="2">
        <parent segment="1"/>
        <proximal x="0.5" y="1" z="0" diameter="1"/>
        <distal x="0.5" y="2" z="0" diameter="2"/>
    </segment>
    <segment id="3">
        <parent segment="1" fractionAlong="0"/>
        <distal x="0.5" y="0" z="3" diameter="2"/>
    </segment>
    <!-- paths and subTrees are essentially equivalent -->
    <segmentGroup id="path01">
        <path>
            <from segment="0"/>
            <to segment="1"/>
        </path>
    </segmentGroup>
    <segmentGroup id="path12">
        <path>
            <from segment="1"/>
            <to segment="2"/>
        </path>
    </segmentGroup>
    <segmentGroup id="path10">
        <path>
            <from segment="1"/>
            <to segment="0"/>
        </path>
    </segmentGroup>
    <segmentGroup id="path0-">
        <path>
            <from segment="0"/>
        </path>
    </segmentGroup>
    <segmentGroup id="path1-">
        <path>
            <from segment="1"/>
        </path>
    </segmentGroup>
    <segmentGroup id="path-3">
        <path>
            <to segment="3"/>
        </path>
    </segmentGroup>
    <segmentGroup id="subTree01">
        <subTree>
            <from segment="0"/>
            <to segment="1"/>
        </subTree>
    </segmentGroup>
    <segmentGroup id="subTree12">
        <subTree>
            <from segment="1"/>
            <to segment="2"/>
        </subTree>
    </segmentGroup>
    <segmentGroup id="subTree10">
        <subTree>
            <from segment="1"/>
            <to segment="0"/>
        </subTree>
    </segmentGroup>
    <segmentGroup id="subTree0-">
        <subTree>
            <from segment="0"/>
        </subTree>
    </segmentGroup>
    <segmentGroup id="subTree1-">
        <subTree>
            <from segment="1"/>
        </subTree>
    </segmentGroup>
    <segmentGroup id="subTree-3">
        <subTree>
            <to segment="3"/>
        </subTree>
    </segmentGroup>
</morphology>
</neuroml>
)~";

    arborio::neuroml N(doc);

    arborio::nml_morphology_data m1 = N.morphology("m1").value();
    label_dict labels;
    labels.import(m1.segments);
    labels.import(m1.groups);
    mprovider P(m1.morphology, labels);

    // Note: paths/subTrees respect segment parent–child relationships,
    // not morphological distality.

    using reg::named;

    EXPECT_TRUE(region_eq(P, named("path01"), join(named("0"), named("1"))));
    EXPECT_TRUE(region_eq(P, named("path12"), join(named("1"), named("2"))));
    EXPECT_TRUE(region_eq(P, named("path10"), reg::nil()));
    EXPECT_TRUE(region_eq(P, named("path0-"), reg::all()));
    EXPECT_TRUE(region_eq(P, named("path1-"), join(named("1"), named("2"), named("3"))));
    EXPECT_TRUE(region_eq(P, named("path-3"), join(named("0"), named("1"), named("3"))));

    EXPECT_TRUE(region_eq(P, named("subTree01"), join(named("0"), named("1"))));
    EXPECT_TRUE(region_eq(P, named("subTree12"), join(named("1"), named("2"))));
    EXPECT_TRUE(region_eq(P, named("subTree10"), reg::nil()));
    EXPECT_TRUE(region_eq(P, named("subTree0-"), reg::all()));
    EXPECT_TRUE(region_eq(P, named("subTree1-"), join(named("1"), named("2"), named("3"))));
    EXPECT_TRUE(region_eq(P, named("subTree-3"), join(named("0"), named("1"), named("3"))));
}
