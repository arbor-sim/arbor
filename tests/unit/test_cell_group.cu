#include "../gtest.h"

#include <cell_group.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../test_common_cells.hpp"

using fvm_cell =
    nest::mc::fvm::fvm_multicell<nest::mc::gpu::backend>;

nest::mc::cell make_cell() {
    using namespace nest::mc;

    nest::mc::cell cell = make_cell_ball_and_stick();

    cell.add_detector({0, 0}, 0);
    cell.segment(1)->set_compartments(101);

    return cell;
}

TEST(cell_group, test)
{
    using namespace nest::mc;

    using cell_group_type = cell_group<fvm_cell>;
    auto group = cell_group_type{0, util::singleton_view(make_cell())};

    group.advance(50, 0.01);

    // the model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}
