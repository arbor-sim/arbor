#include "../gtest.h"

#include <arbor/common_types.hpp>
#include <arborenv/gpu_env.hpp>

#include "epoch.hpp"
#include "execution_context.hpp"
#include "fvm_lowered_cell.hpp"
#include "mc_cell_group.hpp"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;

namespace {
    fvm_lowered_cell_ptr lowered_cell() {
        arb::proc_allocation resources;
        resources.gpu_id = arbenv::default_gpu();
        execution_context context(resources);
        return make_fvm_lowered_cell(backend_kind::gpu, context);
    }

    cable_cell make_cell() {
        soma_cell_builder builder(12.6157/2.0);
        builder.add_branch(0, 200, 0.5, 0.5, 101, "dend");
        cable_cell c = builder.make_cell();
        c.paint("soma", "hh");
        c.paint("dend", "pas");
        c.place(mlocation{1,1}, i_clamp{5, 80, 0.3});
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
