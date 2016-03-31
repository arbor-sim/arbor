#include "gtest.h"

#include "../src/cell.hpp"

TEST(run, init)
{
    using namespace nestmc;

    nestmc::cell cell;

    cell.add_soma(18.8);
    auto& props = cell.soma()->properties;

    cell.construct();

    EXPECT_EQ(cell.graph().num_segments(), 1u);
}
