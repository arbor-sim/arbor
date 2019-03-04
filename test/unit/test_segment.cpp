#include <vector>

#include "../gtest.h"

#include <arbor/math.hpp>
#include <arbor/segment.hpp>

using namespace arb;

TEST(segment, kinfs) {
    using ::arb::math::pi;

    {
        auto s = make_segment<soma_segment>(1.0);
        EXPECT_EQ(s->kind(),   section_kind::soma);
    }

    {
        auto s = make_segment<soma_segment>(1.0, point<double>(0., 1., 2.));
        EXPECT_EQ(s->kind(),   section_kind::soma);
    }

    double length = 1./pi<double>;
    double radius = 1.;

    // single cylindrical frustrum
    {
        auto s = make_segment<cable_segment>(section_kind::dendrite, radius, radius, length);
        EXPECT_EQ(s->kind(),   section_kind::dendrite);
    }

    // cable made up of three identical cylindrical frustrums
    {
        auto s =
            make_segment<cable_segment>(
                section_kind::axon,
                std::vector<double>{radius, radius, radius, radius},
                std::vector<double>{length, length, length}
            );

        EXPECT_EQ(s->kind(),   section_kind::axon);
    }

    // single frustrum of length 1 and radii 1 and 2
    // the centre of each end are at the origin (0,0,0) and (0,1,0)
    {
        auto s =
            make_segment<cable_segment>(
                section_kind::dendrite,
                1, 2,
                point<double>(0,0,0), point<double>(0,1,0)
            );

        EXPECT_EQ(s->kind(),   section_kind::dendrite);
    }

    // cable made up of three frustrums
    // that emulate the frustrum from the previous single-frustrum case
    {
        auto s =
            make_segment<cable_segment>(
                section_kind::axon,
                std::vector<double>{1, 1.5, 2},
                std::vector<point<double>>{ {0,0,0}, {0,0.5,0}, {0,1,0} }
            );

        EXPECT_EQ(s->kind(),   section_kind::axon);
    }
}
