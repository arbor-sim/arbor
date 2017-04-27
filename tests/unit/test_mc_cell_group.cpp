#include "../gtest.h"

#include <backends/multicore/fvm.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <mc_cell_group.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"
#include "../test_common_cells.hpp"

using namespace nest::mc;
using fvm_cell = fvm::fvm_multicell<nest::mc::multicore::backend>;

cell make_cell() {
    auto c = make_cell_ball_and_stick();

    c.add_detector({0, 0}, 0);
    c.segment(1)->set_compartments(101);

    return c;
}


TEST(mc_cell_group, get_kind) {
    mc_cell_group<fvm_cell> group{ 0, util::singleton_view(make_cell()) };

    // we are generating a mc_cell_group which should be of the correct type
    EXPECT_EQ(cell_kind::cable1d_neuron, group.get_cell_kind());
}

TEST(mc_cell_group, test) {
    mc_cell_group<fvm_cell> group{0, util::singleton_view(make_cell())};

    group.advance(50, 0.01);

    // the model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}

TEST(mc_cell_group, sources) {
    using cell_group_type = mc_cell_group<fvm_cell>;

    auto cell = make_cell();
    EXPECT_EQ(cell.detectors().size(), 1u);
    // add another detector on the cell to make things more interesting
    cell.add_detector({1, 0.3}, 2.3);

    cell_gid_type first_gid = 37u;
    auto group = cell_group_type{first_gid, util::singleton_view(cell)};

    // expect group sources to be lexicographically sorted by source id
    // with gids in cell group's range and indices starting from zero

    const auto& sources = group.spike_sources();
    for (unsigned i = 0; i<sources.size(); ++i) {
        auto id = sources[i];
        if (i==0) {
            EXPECT_EQ(id.gid, first_gid);
            EXPECT_EQ(id.index, 0u);
        }
        else {
            auto prev = sources[i-1];
            EXPECT_GT(id, prev);
            EXPECT_EQ(id.index, id.gid==prev.gid? prev.index+1: 0u);
        }
    }
}
