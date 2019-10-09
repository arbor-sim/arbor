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
        auto builder = soma_cell_builder(12.6157/2.0);
        builder.add_dendrite(0, 200, 0.5, 0.5, 101);
        builder.add_stim(mlocation{1,1}, i_clamp{5, 80, 0.3});
        cable_cell c = builder.make_cell();
        c.place(mlocation{0, 0}, threshold_detector{0});
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
