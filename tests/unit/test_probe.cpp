#include "gtest.h"

#include "common_types.hpp"
#include "cell.hpp"
#include "fvm_cell.hpp"

TEST(probe, instantiation)
{
    using namespace nest::mc;

    cell c1;

    segment_location loc1{0, 0};
    segment_location loc2{1, 0.6};

    auto p1 = c1.add_probe({loc1, probeKind::membrane_voltage});
    auto p2 = c1.add_probe({loc2, probeKind::membrane_current});

    // expect locally provided probe ids to be numbered sequentially from zero.

    EXPECT_EQ(0u, p1);
    EXPECT_EQ(1u, p2);

    // expect the probes() return to be a collection with these two probes.

    auto probes = c1.probes();
    EXPECT_EQ(2u, probes.size());

    EXPECT_EQ(loc1, probes[0].location);
    EXPECT_EQ(probeKind::membrane_voltage, probes[0].kind);

    EXPECT_EQ(loc2, probes[1].location);
    EXPECT_EQ(probeKind::membrane_current, probes[1].kind);
}

TEST(probe, fvm_cell)
{
    using namespace nest::mc;

    cell bs;

    // ball-and-stick model morphology

    bs.add_soma(12.6157/2.0);
    bs.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);
    bs.soma()->set_compartments(5);

    segment_location loc0{0, 0};
    segment_location loc1{1, 1};
    segment_location loc2{1, 0.5};

    auto pv0 = bs.add_probe({loc0, probeKind::membrane_voltage});
    auto pv1 = bs.add_probe({loc1, probeKind::membrane_voltage});
    auto pi2 = bs.add_probe({loc2, probeKind::membrane_current});

    i_clamp stim(0, 100, 0.3);
    bs.add_stimulus({1, 1}, stim);

    fvm::fvm_cell<double, cell_local_size_type> lcell(bs);
    lcell.setup_matrix(0.01);
    lcell.initialize();

    EXPECT_EQ(3u, lcell.num_probes());

    // expect probe values and direct queries of voltage and current
    // to be equal in fvm cell

    EXPECT_EQ(lcell.voltage(loc0), lcell.probe(pv0));
    EXPECT_EQ(lcell.voltage(loc1), lcell.probe(pv1));
    EXPECT_EQ(lcell.current(loc2), lcell.probe(pi2));

    lcell.advance(0.05);

    EXPECT_EQ(lcell.voltage(loc0), lcell.probe(pv0));
    EXPECT_EQ(lcell.voltage(loc1), lcell.probe(pv1));
    EXPECT_EQ(lcell.current(loc2), lcell.probe(pi2));
}


