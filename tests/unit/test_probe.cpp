#include "gtest.h"

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_cell.hpp>
#include <util/singleton.hpp>

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

    bs.add_probe({loc0, probeKind::membrane_voltage});
    bs.add_probe({loc1, probeKind::membrane_voltage});
    bs.add_probe({loc2, probeKind::membrane_current});

    i_clamp stim(0, 100, 0.3);
    bs.add_stimulus({1, 1}, stim);

    using fvm_cell = fvm::fvm_cell<double, cell_lid_type>;
    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes{3};

    fvm_cell lcell;
    lcell.initialize(util::singleton_view(bs), detectors, targets, probes);

    // Know from implementation that probe_handle.second
    // is a compartment index: expect probe values and
    // direct queries of voltage and current
    // to be equal in fvm cell

    EXPECT_EQ(lcell.voltage()[probes[0].second], lcell.probe(probes[0]));
    EXPECT_EQ(lcell.voltage()[probes[1].second], lcell.probe(probes[1]));
    EXPECT_EQ(lcell.current()[probes[2].second], lcell.probe(probes[2]));

    lcell.advance(0.05);

    EXPECT_EQ(lcell.voltage()[probes[0].second], lcell.probe(probes[0]));
    EXPECT_EQ(lcell.voltage()[probes[1].second], lcell.probe(probes[1]));
    EXPECT_EQ(lcell.current()[probes[2].second], lcell.probe(probes[2]));
}


