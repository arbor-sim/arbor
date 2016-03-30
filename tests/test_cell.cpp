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
        c.add_cable(std::move(seg), 0);

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
