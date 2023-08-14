#include <gtest/gtest.h>

#include <arbor/common_types.hpp>
#include <arborio/label_parse.hpp>
#include <arborenv/default_env.hpp>

#include "epoch.hpp"
#include "execution_context.hpp"
#include "fvm_lowered_cell.hpp"
#include "mc_cell_group.hpp"

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using namespace arborio::literals;

namespace {
    cable_cell_description make_cell() {
        soma_cell_builder builder(12.6157/2.0);
        builder.add_branch(0, 200, 0.5, 0.5, 101, "dend");
        auto d = builder.make_cell();
        d.decorations.paint("soma"_lab, density("hh"));
        d.decorations.paint("dend"_lab, density("pas"));
        d.decorations.place(builder.location({1,1}), i_clamp::box(5, 80, 0.3), "clamp0");
        d.decorations.place(builder.location({0, 0}), threshold_detector{0}, "detector0");
        return d;
    }
}

TEST(mc_cell_group, gpu_test)
{
    auto context = make_context({1, arbenv::default_gpu()});

    cable_cell cell = make_cell();
    auto rec = cable1d_recipe({cell});
    rec.nernst_ion("na");
    rec.nernst_ion("ca");
    rec.nernst_ion("k");

    cell_label_range srcs, tgts;
    mc_cell_group group{{0}, rec, srcs, tgts, make_fvm_lowered_cell(backend_kind::gpu, *context)};
    group.advance(epoch(0, 0., 50.), 0.01, {});

    // The model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms
    EXPECT_EQ(4u, group.spikes().size());
}
