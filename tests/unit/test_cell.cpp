#include "../gtest.h"

#include "cell.hpp"

TEST(cell, soma)
{
    // test that insertion of a soma works
    //      define with no centre point
    {
        arb::cell c;
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
        arb::cell c;
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

TEST(cell, add_segment)
{
    using namespace arb;
    //  add a pre-defined segment
    {
        cell c;

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
        cell c;

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
        cell c;

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

TEST(cell, multiple_cables)
{
    using namespace arb;

    // generate a cylindrical cable segment of length 1/pi and radius 1
    //      volume = 1
    //      area   = 2
    auto seg = [](section_kind k) {
        return make_segment<cable_segment>( k, 1.0, 1.0, 1./math::pi<double>() );
    };

    //  add a pre-defined segment
    {
        cell c;

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
        // each of the 5 segments has volume 1 by design
        EXPECT_EQ(c.volume(), 5.);
        // each of the 4 cables has volume 2., and the soma has an awkward area
        // that isn't a round number
        EXPECT_EQ(c.area(), 8. + math::area_sphere(soma_radius));

        // construct the graph
        const auto model = c.model();
        auto const& con = model.tree;

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

TEST(cell, unbranched_chain)
{
    using namespace arb;

    cell c;

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
    // each of the 3 segments has volume 1 by design
    EXPECT_EQ(c.volume(), 3.);
    // each of the 2 cables has volume 2., and the soma has an awkward area
    // that isn't a round number
    EXPECT_EQ(c.area(), 4. + math::area_sphere(soma_radius));

    // construct the graph
    const auto tree = c.model().tree;

    auto no_parent = tree::no_parent;
    EXPECT_EQ(tree.num_segments(), 3u);
    EXPECT_EQ(tree.parent(0), no_parent);
    EXPECT_EQ(tree.parent(1), 0u);
    EXPECT_EQ(tree.parent(2), 1u);
    EXPECT_EQ(tree.num_children(0), 1u);
    EXPECT_EQ(tree.num_children(1), 1u);
    EXPECT_EQ(tree.num_children(2), 0u);
}

TEST(cell, clone)
{
    using namespace arb;

    // make simple cell with multiple segments

    cell c;
    c.add_soma(2.1);
    c.add_cable(0, section_kind::dendrite, 0.3, 0.2, 10);
    c.segment(1)->set_compartments(3);
    c.add_cable(1, section_kind::dendrite, 0.2, 0.15, 20);
    c.segment(2)->set_compartments(5);

    c.add_synapse({1, 0.3}, mechanism_spec("expsyn"));

    c.add_detector({0, 0.5}, 10.0);

    // make clone

    cell d(c);

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

TEST(cell, get_kind)
{
    using namespace arb;

    // make a MC cell
    cell c;
    EXPECT_EQ( cell_kind::cable1d_neuron, c.get_cell_kind());
}
