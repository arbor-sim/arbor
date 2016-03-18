#include <limits>

#include "gtest.h"

#include "../src/segment.hpp"

TEST(segments, sphere)
{
    using namespace nestmc;

    {
        auto s = make_segment<spherical_segment>(1.0);

        EXPECT_EQ(s->volume(), nestmc::pi<double>()*4./3.);
        EXPECT_EQ(s->area(),   nestmc::pi<double>()*4.);
        EXPECT_EQ(s->kind(),   nestmc::segmentKind::soma);
    }

    {
        auto s = make_segment<spherical_segment>(1.0, point<double>(0., 1., 2.));

        EXPECT_EQ(s->volume(), nestmc::pi<double>()*4./3.);
        EXPECT_EQ(s->area(),   nestmc::pi<double>()*4.);
        EXPECT_EQ(s->kind(),   nestmc::segmentKind::soma);
    }
}

TEST(segments, frustrum)
{
    {
    }
}
