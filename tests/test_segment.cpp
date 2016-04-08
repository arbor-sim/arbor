#include <limits>

#include "gtest.h"

#include "../src/segment.hpp"


TEST(segments, soma)
{
    using namespace nest::mc;
    using nest::mc::math::pi;

    {
        auto s = make_segment<soma_segment>(1.0);

        EXPECT_EQ(s->volume(), pi<double>()*4./3.);
        EXPECT_EQ(s->area(),   pi<double>()*4.);
        EXPECT_EQ(s->kind(),   segmentKind::soma);
    }

    {
        auto s = make_segment<soma_segment>(1.0, point<double>(0., 1., 2.));

        EXPECT_EQ(s->volume(), pi<double>()*4./3.);
        EXPECT_EQ(s->area(),   pi<double>()*4.);
        EXPECT_EQ(s->kind(),   segmentKind::soma);
    }
}

TEST(segments, cable)
{
    using namespace nest::mc;
    using nest::mc::math::pi;

    // take advantage of fact that a cable segment with constant radius 1 and
    // length 1 has volume=1. and area=2
    auto length = 1./pi<double>();
    auto radius = 1.;

    // single cylindrical frustrum
    {
        auto s = make_segment<cable_segment>(segmentKind::dendrite, radius, radius, length);

        EXPECT_EQ(s->volume(), 1.0);
        EXPECT_EQ(s->area(),   2.0);
        EXPECT_EQ(s->kind(),   segmentKind::dendrite);
    }

    // cable made up of three identical cylindrical frustrums
    {
        auto s =
            make_segment<cable_segment>(
                segmentKind::axon,
                std::vector<double>{radius, radius, radius, radius},
                std::vector<double>{length, length, length}
            );

        EXPECT_EQ(s->volume(), 3.0);
        EXPECT_EQ(s->area(),   6.0);
        EXPECT_EQ(s->kind(),   segmentKind::axon);
    }
}

TEST(segments, cable_positions)
{
    using namespace nest::mc;
    using nest::mc::math::pi;

    // single frustrum of length 1 and radii 1 and 2
    // the centre of each end are at the origin (0,0,0) and (0,1,0)
    {
        auto s =
            make_segment<cable_segment>(
                segmentKind::dendrite,
                1, 2,
                point<double>(0,0,0), point<double>(0,1,0)
            );

        EXPECT_EQ(s->volume(), math::volume_frustrum(1., 1., 2.));
        EXPECT_EQ(s->area(),   math::area_frustrum  (1., 1., 2.));
        EXPECT_EQ(s->kind(),   segmentKind::dendrite);
    }

    // cable made up of three frustrums
    // that emulate the frustrum from the previous single-frustrum case
    {
        auto s =
            make_segment<cable_segment>(
                segmentKind::axon,
                std::vector<double>{1, 1.5, 2},
                std::vector<point<double>>{ {0,0,0}, {0,0.5,0}, {0,1,0} }
            );

        EXPECT_EQ(s->volume(), math::volume_frustrum(1., 1., 2.));
        EXPECT_EQ(s->area(),   math::area_frustrum(1., 1., 2.));
        EXPECT_EQ(s->kind(),   segmentKind::axon);
    }
}
