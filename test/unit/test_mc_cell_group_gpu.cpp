#include "../gtest.h"

#include <arbor/common_types.hpp>

#include "epoch.hpp"
#include "execution_context.hpp"
#include "fvm_lowered_cell.hpp"
#include "mc_cell_group.hpp"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;

namespace {
    fvm_lowered_cell_ptr lowered_cell() {
        execution_context context;
        return make_fvm_lowered_cell(backend_kind::gpu, context);
    }

    cable_cell make_cell() {
        auto c = make_cell_ball_and_stick();

        c.add_detector({0, 0}, 0);
        c.segment(1)->set_compartments(101);

        return c;
    }
}

TEST(mc_cell_group, gpu_test)
{
    auto rec = cable1d_recipe(make_cell());
    rec.nernst_ion("na");
    rec.nernst_ion("ca");
    rec.nernst_ion("k");

    mc_cell_group group{{0}, rec, lowered_cell()};
    group.advance(epoch(0, 50), 0.01, {});

    // The model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}
