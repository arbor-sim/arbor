#include "../gtest.h"

#include <backends/gpu/fvm.hpp>
#include <mc_cell_group.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../test_common_cells.hpp"

using namespace nest::mc;

using fvm_cell =
    fvm::fvm_multicell<nest::mc::gpu::backend>;

nest::mc::cell make_cell() {
    using namespace nest::mc;

    cell c = make_cell_ball_and_stick();

    c.add_detector({0, 0}, 0);
    c.segment(1)->set_compartments(101);

    return c;
}

TEST(cell_group, test)
{
    using cell_group_type = mc_cell_group<fvm_cell>;
    auto group = cell_group_type({0u}, util::singleton_view(make_cell()));

    group.advance(50, 0.01);

    // the model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}
