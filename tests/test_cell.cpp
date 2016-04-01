#include "gtest.h"

#include "../src/cell.hpp"

TEST(cell_type, soma)
{
    // test that insertion of a soma works
    //      define with no centre point
    {
        nestmc::cell c;
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
        nestmc::cell c;
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

TEST(cell_type, add_segment)
{
    using namespace nestmc;
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
                segmentKind::dendrite,
                cable_radius, cable_radius, cable_length
            );
        c.add_cable(0, std::move(seg));

        EXPECT_EQ(c.num_segments(), 2);
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
            segmentKind::dendrite, cable_radius, cable_radius, cable_length
        );

        EXPECT_EQ(c.num_segments(), 2);
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
            segmentKind::dendrite,
            std::vector<double>{cable_radius, cable_radius, cable_radius, cable_radius},
            std::vector<double>{cable_length, cable_length, cable_length}
        );

        EXPECT_EQ(c.num_segments(), 2);
    }
}

TEST(cell_type, multiple_cables)
{
    using namespace nestmc;

    // generate a cylindrical cable segment of length 1/pi and radius 1
    //      volume = 1
    //      area   = 2
    auto seg = [](segmentKind k) {
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
        c.add_cable(0, seg(segmentKind::dendrite));
        c.add_cable(0, seg(segmentKind::axon));
        c.add_cable(1, seg(segmentKind::dendrite));
        c.add_cable(1, seg(segmentKind::dendrite));

        EXPECT_EQ(c.num_segments(), 5);
        // each of the 5 segments has volume 1 by design
        EXPECT_EQ(c.volume(), 5.);
        // each of the 4 cables has volume 2., and the soma has an awkward area
        // that isn't a round number
        EXPECT_EQ(c.area(), 8. + math::area_sphere(soma_radius));

        // construct the graph
        auto const& con = c.graph();

        EXPECT_EQ(con.num_segments(), 5u);
        EXPECT_EQ(con.parent(0), -1);
        EXPECT_EQ(con.parent(1), 0);
        EXPECT_EQ(con.parent(2), 0);
        EXPECT_EQ(con.parent(3), 1);
        EXPECT_EQ(con.parent(4), 1);
        EXPECT_EQ(con.num_children(0), 2u);
        EXPECT_EQ(con.num_children(1), 2u);
        EXPECT_EQ(con.num_children(2), 0u);
        EXPECT_EQ(con.num_children(3), 0u);
        EXPECT_EQ(con.num_children(4), 0u);

    }
}

