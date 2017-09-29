#include "../gtest.h"

#include <backends/gpu/fvm.hpp>
#include <mc_cell_group.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using fvm_cell = fvm::fvm_multicell<arb::gpu::backend>;

cell make_cell() {
    auto c = make_cell_ball_and_stick();

    c.add_detector({0, 0}, 0);
    c.segment(1)->set_compartments(101);

    return c;
}

TEST(mc_cell_group, test)
{
    mc_cell_group<fvm_cell> group({0u}, cable1d_recipe(make_cell()));

    group.advance(50, 0.01);

    // the model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}
