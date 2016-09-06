#include <fstream>

#include "gtest.h"

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <util/range.hpp>

#include "../test_util.hpp"
#include "../test_common_cells.hpp"

TEST(fvm_multi, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell=make_cell_ball_and_3sticks();

    using fvm_cell = fvm::fvm_multicell<double, cell_lid_type>;

    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), detectors, targets, probes);

    auto& J = fvcell.jacobian();

    // 1 (soma) + 3 (dendritic segments) × 4 compartments
    EXPECT_EQ(cell.num_compartments(), 13u);

    // assert that the matrix has one row for each compartment
    EXPECT_EQ(J.size(), cell.num_compartments());

    fvcell.setup_matrix(0.02);

    // assert that the number of cv areas is the same as the matrix size
    // i.e. both should equal the number of compartments
    EXPECT_EQ(fvcell.cv_areas().size(), J.size());
}

TEST(fvm_multi, init)
{
    using namespace nest::mc;

    nest::mc::cell cell = make_cell_ball_and_stick();

    const auto m = cell.model();
    EXPECT_EQ(m.tree.num_segments(), 2u);

    auto& soma_hh = cell.soma()->mechanism("hh");

    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.3);

    // check that parameter values were set correctly
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gnabar").value, 0.12);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gkbar").value, 0.036);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("gl").value, 0.0003);
    EXPECT_EQ(cell.soma()->mechanism("hh").get("el").value, -54.3);

    cell.segment(1)->set_compartments(10);

    using fvm_cell = fvm::fvm_multicell<double, cell_lid_type>;
    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), detectors, targets, probes);

    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    fvcell.setup_matrix(0.01);
}

TEST(fvm_multi, multi_init)
{
    using namespace nest::mc;

    nest::mc::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_3sticks()
    };

    EXPECT_EQ(cells[0].num_segments(), 2u);
    EXPECT_EQ(cells[0].segment(1)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].num_segments(), 4u);
    EXPECT_EQ(cells[1].segment(1)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].segment(2)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].segment(3)->num_compartments(), 4u);

    cells[0].add_synapse({1, 0.4}, parameter_list("expsyn"));
    cells[0].add_synapse({1, 0.4}, parameter_list("expsyn"));
    cells[1].add_synapse({2, 0.4}, parameter_list("exp2syn"));
    cells[1].add_synapse({3, 0.4}, parameter_list("expsyn"));

    cells[1].add_detector({0, 0}, 3.3);

    using fvm_cell = fvm::fvm_multicell<double, cell_lid_type>;
    std::vector<fvm_cell::target_handle> targets(4);
    std::vector<fvm_cell::detector_handle> detectors(1);
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(cells, detectors, targets, probes);

    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 5u+13u);

    // check indices in instantiated mechanisms
    for (const auto& mech: fvcell.mechanisms()) {
        if (mech->name()=="hh") {
            // HH on somas of two cells, with group compartment indices
            // 0 and 5.
            ASSERT_EQ(mech->node_index().size(), 2u);
            EXPECT_EQ(mech->node_index()[0], 0u);
            EXPECT_EQ(mech->node_index()[1], 5u);
        }
        if (mech->name()=="expsyn") {
            // Three expsyn synapses, two in second compartment
            // of dendrite segment of first cell, one in second compartment
            // of last segment of second cell.
            ASSERT_EQ(mech->node_index().size(), 3u);
            EXPECT_EQ(mech->node_index()[0], 2u);
            EXPECT_EQ(mech->node_index()[1], 2u);
            EXPECT_EQ(mech->node_index()[2], 15u);
        }
        if (mech->name()=="exp2syn") {
            // One exp2syn synapse, in second compartment
            // of penultimate segment of second cell.
            ASSERT_EQ(mech->node_index().size(), 1u);
            EXPECT_EQ(mech->node_index()[0], 11u);
        }
    }

    fvcell.setup_matrix(0.01);
}
