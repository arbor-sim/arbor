#include <fstream>

#include "gtest.h"

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_cell.hpp>
#include <util/rangeutil.hpp>

#include "../test_common_cells.hpp"
#include "../test_util.hpp"

TEST(fvm, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell = make_cell_ball_and_3stick();

    using fvm_cell = fvm::fvm_cell<double, cell_lid_type>;

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

TEST(fvm, init)
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

    using fvm_cell = fvm::fvm_cell<double, cell_lid_type>;
    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), detectors, targets, probes);

    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    fvcell.setup_matrix(0.01);
}
