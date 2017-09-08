#include "../gtest.h"

#include <backends/multicore/fvm.hpp>
#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace nest::mc;

TEST(probe, fvm_multicell)
{
    using fvm_cell = fvm::fvm_multicell<nest::mc::multicore::backend>;

    cell bs = make_cell_ball_and_stick(false);

    i_clamp stim(0, 100, 0.3);
    bs.add_stimulus({1, 1}, stim);

    cable1d_recipe rec(bs);

    segment_location loc0{0, 0};
    segment_location loc1{1, 1};
    segment_location loc2{1, 0.7};

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

    // Know from implementation that probe_handle.second
    // is a compartment index: expect probe values and
    // direct queries of voltage and current to be equal in fvm_cell.

    EXPECT_EQ(lcell.voltage()[p0.second], lcell.probe(p0));
    EXPECT_EQ(lcell.voltage()[p1.second], lcell.probe(p1));
    EXPECT_EQ(lcell.current()[p2.second], lcell.probe(p2));

    lcell.setup_integration(0.05, 0.05);
    lcell.step_integration();

    EXPECT_EQ(lcell.voltage()[p0.second], lcell.probe(p0));
    EXPECT_EQ(lcell.voltage()[p1.second], lcell.probe(p1));
    EXPECT_EQ(lcell.current()[p2.second], lcell.probe(p2));
}

