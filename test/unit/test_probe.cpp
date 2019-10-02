#include "../gtest.h"

#include <arbor/common_types.hpp>
#include <arbor/cable_cell.hpp>

#include "backends/event.hpp"
#include "backends/multicore/fvm.hpp"
#include "fvm_lowered_cell_impl.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;
using fvm_cell = fvm_lowered_cell_impl<multicore::backend>;
using shared_state = multicore::backend::shared_state;

ACCESS_BIND(std::unique_ptr<shared_state> fvm_cell::*, fvm_state_ptr, &fvm_cell::state_);

TEST(probe, fvm_lowered_cell) {
    execution_context context;

    cable_cell bs = make_cell_ball_and_stick(false);

    i_clamp stim(0, 100, 0.3);
    bs.place(mlocation{1, 1}, stim);

    cable1d_recipe rec(bs);

    mlocation loc0{0, 0};
    mlocation loc1{1, 1};
    mlocation loc2{1, 0.3};

    rec.add_probe(0, 10, cell_probe_address{loc0, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 20, cell_probe_address{loc1, cell_probe_address::membrane_voltage});
    rec.add_probe(0, 30, cell_probe_address{loc2, cell_probe_address::membrane_current});

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(context);
    lcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    EXPECT_EQ(3u, rec.num_probes(0));
    EXPECT_EQ(3u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({0, 1}).tag);
    EXPECT_EQ(30, probe_map.at({0, 2}).tag);

    probe_handle p0 = probe_map.at({0, 0}).handle;
    probe_handle p1 = probe_map.at({0, 1}).handle;
    probe_handle p2 = probe_map.at({0, 2}).handle;

    // Expect initial probe values to be the resting potential
    // for the voltage probes (cell membrane potential should
    // be constant), and zero for the current probe.

    auto& state = *(lcell.*fvm_state_ptr).get();
    auto& voltage = state.voltage;

    auto resting = voltage[0];
    EXPECT_NE(0.0, resting);

    // (Probe handles are just pointers in this implementation).
    EXPECT_EQ(resting, *p0);
    EXPECT_EQ(resting, *p1);
    EXPECT_EQ(0.0, *p2);

    // After an integration step, expect voltage probe values
    // to differ from resting, and between each other, and
    // for there to be a non-zero current.
    //
    // First probe, at (0,0), should match voltage in first
    // compartment.

    lcell.integrate(0.01, 0.0025, {}, {});

    EXPECT_NE(resting, *p0);
    EXPECT_NE(resting, *p1);
    EXPECT_NE(*p0, *p1);
    EXPECT_NE(0.0, *p2);

    EXPECT_EQ(voltage[0], *p0);
}

