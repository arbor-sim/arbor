#include "../gtest.h"

#include <arbor/mc_cell.hpp>

#include "math.hpp"
#include "tree.hpp"

using namespace arb;

TEST(mc_cell, soma) {
    // test that insertion of a soma works
    //      define with no centre point
    {
        mc_cell c;
        auto soma_radius = 2.1;

        EXPECT_EQ(c.has_soma(), false);
        c.add_soma(soma_radius);
        EXPECT_EQ(c.has_soma(), true);

        auto s = c.soma();
        EXPECT_EQ(s->radius(), soma_radius);
        EXPECT_EQ(s->center().is_set(), false);
    }

    // test that insertion of a soma works
    //      define with centre point @ (0,0,1)
    {
        mc_cell c;
        auto soma_radius = 3.2;

        EXPECT_EQ(c.has_soma(), false);
        c.add_soma(soma_radius, {0,0,1});
        EXPECT_EQ(c.has_soma(), true);

        auto s = c.soma();
        EXPECT_EQ(s->radius(), soma_radius);
        EXPECT_EQ(s->center().is_set(), true);

        // add expression template library for points
        //EXPECT_EQ(s->center(), point<double>(0,0,1));
    }
}

TEST(mc_cell, add_segment) {
    //  add a pre-defined segment
    {
        mc_cell c;

        auto soma_radius  = 2.1;
        auto cable_radius = 0.1;
        auto cable_length = 8.3;

        // add a soma because we need something to attach the first dendrite to
        c.add_soma(soma_radius, {0,0,1});

        auto seg =
            make_segment<cable_segment>(
                section_kind::dendrite,
                cable_radius, cable_radius, cable_length
            );
        c.add_cable(0, std::move(seg));

        EXPECT_EQ(c.num_segments(), 2u);
    }

    //  add segment on the fly
    {
        mc_cell c;

        auto soma_radius  = 2.1;
        auto cable_radius = 0.1;
        auto cable_length = 8.3;

        // add a soma because we need something to attach the first dendrite to
        c.add_soma(soma_radius, {0,0,1});

        c.add_cable(
            0,
            section_kind::dendrite, cable_radius, cable_radius, cable_length
        );

        EXPECT_EQ(c.num_segments(), 2u);
    }
    {
        mc_cell c;

        auto soma_radius  = 2.1;
        auto cable_radius = 0.1;
        auto cable_length = 8.3;

        // add a soma because we need something to attach the first dendrite to
        c.add_soma(soma_radius, {0,0,1});

        c.add_cable(
            0,
            section_kind::dendrite,
            std::vector<double>{cable_radius, cable_radius, cable_radius, cable_radius},
            std::vector<double>{cable_length, cable_length, cable_length}
        );

        EXPECT_EQ(c.num_segments(), 2u);
    }
}

TEST(mc_cell, multiple_cables) {
    // generate a cylindrical cable segment of length 1/pi and radius 1
    //      volume = 1
    //      area   = 2
    auto seg = [](section_kind k) {
        return make_segment<cable_segment>( k, 1.0, 1.0, 1./math::pi<double>() );
    };

    //  add a pre-defined segment
    {
        mc_cell c;

        auto soma_radius = std::pow(3./(4.*math::pi<double>()), 1./3.);

        // cell strucure as follows
        // left   :  segment numbering
        // right  :  segment type (soma, axon, dendrite)
        //
        //          0           s
        //         / \         / \.
        //        1   2       d   a
        //       / \         / \.
        //      3   4       d   d

        // add a soma
        c.add_soma(soma_radius, {0,0,1});

        // hook the dendrite and axons
        c.add_cable(0, seg(section_kind::dendrite));
        c.add_cable(0, seg(section_kind::axon));
        c.add_cable(1, seg(section_kind::dendrite));
        c.add_cable(1, seg(section_kind::dendrite));

        EXPECT_EQ(c.num_segments(), 5u);

        // construct the graph
        tree con(c.parents());

        auto no_parent = tree::no_parent;
        EXPECT_EQ(con.num_segments(), 5u);
        EXPECT_EQ(con.parent(0), no_parent);
        EXPECT_EQ(con.parent(1), 0u);
        EXPECT_EQ(con.parent(2), 0u);
        EXPECT_EQ(con.parent(3), 1u);
        EXPECT_EQ(con.parent(4), 1u);
        EXPECT_EQ(con.num_children(0), 2u);
        EXPECT_EQ(con.num_children(1), 2u);
        EXPECT_EQ(con.num_children(2), 0u);
        EXPECT_EQ(con.num_children(3), 0u);
        EXPECT_EQ(con.num_children(4), 0u);

    }
}

TEST(mc_cell, unbranched_chain) {
    mc_cell c;

    auto soma_radius = std::pow(3./(4.*math::pi<double>()), 1./3.);

    // Cell strucure that looks like a centipede: i.e. each segment has only one child
    //
    //   |       |
    //  0|1-2-3-4|5-6-7-8
    //   |       |

    // add a soma
    c.add_soma(soma_radius, {0,0,1});

    // hook the dendrite and axons
    c.add_cable(0, make_segment<cable_segment>(section_kind::dendrite, 1.0, 1.0, 1./math::pi<double>()));
    c.add_cable(1, make_segment<cable_segment>(section_kind::dendrite, 1.0, 1.0, 1./math::pi<double>()));

    EXPECT_EQ(c.num_segments(), 3u);

    // construct the graph
    tree con(c.parents());

    auto no_parent = tree::no_parent;
    EXPECT_EQ(con.num_segments(), 3u);
    EXPECT_EQ(con.parent(0), no_parent);
    EXPECT_EQ(con.parent(1), 0u);
    EXPECT_EQ(con.parent(2), 1u);
    EXPECT_EQ(con.num_children(0), 1u);
    EXPECT_EQ(con.num_children(1), 1u);
    EXPECT_EQ(con.num_children(2), 0u);
}

TEST(mc_cell, clone) {
    // make simple cell with multiple segments

    mc_cell c;
    c.add_soma(2.1);
    c.add_cable(0, section_kind::dendrite, 0.3, 0.2, 10);
    c.segment(1)->set_compartments(3);
    c.add_cable(1, section_kind::dendrite, 0.2, 0.15, 20);
    c.segment(2)->set_compartments(5);

    c.add_synapse({1, 0.3}, "expsyn");

    c.add_detector({0, 0.5}, 10.0);

    // make clone

    mc_cell d(c);

    // check equality

    ASSERT_EQ(c.num_segments(), d.num_segments());
    EXPECT_EQ(c.soma()->radius(), d.soma()->radius());
    EXPECT_EQ(c.segment(1)->as_cable()->length(), d.segment(1)->as_cable()->length());
    {
        const auto& csyns = c.synapses();
        const auto& dsyns = d.synapses();

        ASSERT_EQ(csyns.size(), dsyns.size());
        for (unsigned i=0; i<csyns.size(); ++i) {
            ASSERT_EQ(csyns[i].location, dsyns[i].location);
        }
    }

    ASSERT_EQ(1u, c.detectors().size());
    ASSERT_EQ(1u, d.detectors().size());
    EXPECT_EQ(c.detectors()[0].threshold, d.detectors()[0].threshold);

    // check clone is independent

    c.add_cable(2, section_kind::dendrite, 0.15, 0.1, 20);
    EXPECT_NE(c.num_segments(), d.num_segments());

    d.detectors()[0].threshold = 13.0;
    ASSERT_EQ(1u, c.detectors().size());
    ASSERT_EQ(1u, d.detectors().size());
    EXPECT_NE(c.detectors()[0].threshold, d.detectors()[0].threshold);

    c.segment(1)->set_compartments(7);
    EXPECT_NE(c.segment(1)->num_compartments(), d.segment(1)->num_compartments());
    EXPECT_EQ(c.segment(2)->num_compartments(), d.segment(2)->num_compartments());
}

TEST(mc_cell, get_kind) {
    mc_cell c;
    EXPECT_EQ(cell_kind::cable1d_neuron, c.get_cell_kind());
}
