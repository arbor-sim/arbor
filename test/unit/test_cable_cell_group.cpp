#include <gtest/gtest.h>

#include <arbor/common_types.hpp>

#include <arborio/label_parse.hpp>

#include "epoch.hpp"
#include "fvm_lowered_cell.hpp"
#include "mc_cell_group.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using namespace arborio::literals;

namespace {
    execution_context context;

    fvm_lowered_cell_ptr lowered_cell() {
        return make_fvm_lowered_cell(backend_kind::multicore, context);
    }

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

ACCESS_BIND(
    std::vector<cell_member_type> mc_cell_group::*,
    private_spike_sources_ptr,
    &mc_cell_group::spike_sources_)

TEST(mc_cell_group, get_kind) {
    cable_cell cell = make_cell();
    cell_label_range srcs, tgts;
    mc_cell_group group{{0}, cable1d_recipe({cell}), srcs, tgts, lowered_cell()};

    EXPECT_EQ(cell_kind::cable, group.get_cell_kind());
}

TEST(mc_cell_group, test) {
    cable_cell cell = make_cell();
    auto rec = cable1d_recipe({cell});
    rec.nernst_ion("na");
    rec.nernst_ion("ca");
    rec.nernst_ion("k");

    cell_label_range srcs, tgts;
    mc_cell_group group{{0}, rec, srcs, tgts, lowered_cell()};
    group.advance(epoch(0, 0., 50.), 0.01, {});

    // Model is expected to generate 4 spikes as a result of the
    // fixed stimulus over 50 ms.
    EXPECT_EQ(4u, group.spikes().size());
}

TEST(mc_cell_group, sources) {
    // Make twenty cells, with an extra detector on gids 0, 3 and 17
    // to make things more interesting.
    std::vector<cable_cell> cells;

    for (int i=0; i<20; ++i) {
        auto desc = make_cell();
        if (i==0 || i==3 || i==17) {
            desc.decorations.place(mlocation{0, 0.3}, threshold_detector{2.3}, "detector1");
        }
        cells.emplace_back(desc);

        EXPECT_EQ(1u + (i==0 || i==3 || i==17), cells.back().detectors().size());
    }

    std::vector<cell_gid_type> gids = {3u, 4u, 10u, 16u, 17u, 18u};
    auto rec = cable1d_recipe(cells);
    rec.nernst_ion("na");
    rec.nernst_ion("ca");
    rec.nernst_ion("k");

    cell_label_range srcs, tgts;
    mc_cell_group group{gids, rec, srcs, tgts, lowered_cell()};

    // Expect group sources to be lexicographically sorted by source id
    // with gids in cell group's range and indices starting from zero.

    const auto& sources = group.*private_spike_sources_ptr;
    for (unsigned j = 0; j<sources.size(); ++j) {
        auto id = sources[j];
        if (j==0) {
            EXPECT_EQ(id.gid, gids[0]);
            EXPECT_EQ(id.index, 0u);
        }
        else {
            auto prev = sources[j-1];
            EXPECT_GT(id, prev);
            EXPECT_EQ(id.index, id.gid==prev.gid? prev.index+1: 0u);
        }
    }
}

