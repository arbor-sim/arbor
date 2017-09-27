#include "../gtest.h"

#include <backends/multicore/fvm.hpp>
#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;

TEST(probe, fvm_multicell)
{
    using fvm_cell = fvm::fvm_multicell<arb::multicore::backend>;

    cell bs = make_cell_ball_and_stick(false);

    i_clamp stim(0, 100, 0.3);
    bs.add_stimulus({1, 1}, stim);

    cable1d_recipe rec(bs);

    segment_location loc0{0, 0};
    segment_location loc1{1, 1};
    segment_location loc2{1, 0.3};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 20, cell_probe_address{loc1, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 30, cell_probe_address{loc2, cell_probe_address::membrane_current});

    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell lcell;
    lcell.initialize({0}, rec, targets, probe_map);

    EXPECT_EQ(3u, rec.num_probes(0));
    EXPECT_EQ(3u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({0, 1}).tag);
    EXPECT_EQ(30, probe_map.at({0, 2}).tag);

    fvm_cell::probe_handle p0 = probe_map.at({0, 0}).handle;
    fvm_cell::probe_handle p1 = probe_map.at({0, 1}).handle;
    fvm_cell::probe_handle p2 = probe_map.at({0, 2}).handle;

    // Expect initial probe values to be the resting potential
    // for the voltage probes (cell membrane potential should
    // be constant), and zero for the current probe.

    auto resting = lcell.voltage()[0];
    EXPECT_NE(0.0, resting);

    EXPECT_EQ(resting, lcell.probe(p0));
    EXPECT_EQ(resting, lcell.probe(p1));
    EXPECT_EQ(0.0, lcell.probe(p2));

    // After an integration step, expect voltage probe values
    // to differ from resting, and between each other, and
    // for there to be a non-zero current.
    //
    // First probe, at (0,0), should match voltage in first
    // compartment.

    lcell.setup_integration(0.1, 0.0025, {}, {});
    lcell.step_integration();
    lcell.step_integration();

    EXPECT_NE(resting, lcell.probe(p0));
    EXPECT_NE(resting, lcell.probe(p1));
    EXPECT_NE(lcell.probe(p0), lcell.probe(p1));
    EXPECT_NE(0.0, lcell.probe(p2));

    EXPECT_EQ(lcell.voltage()[0], lcell.probe(p0));
}

