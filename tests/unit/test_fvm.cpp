#include <fstream>

#include "gtest.h"
#include "../test_util.hpp"

#include <cell.hpp>
#include <fvm_cell.hpp>

TEST(fvm, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    cell.add_soma(6e-4); // 6um in cm

    // 1um radius and 4mm long, all in cm
    cell.add_cable(0, segmentKind::dendrite, 1e-4, 1e-4, 4e-1);
    cell.add_cable(0, segmentKind::dendrite, 1e-4, 1e-4, 4e-1);

    //std::cout << cell.segment(1)->area() << " is the area\n";
    EXPECT_EQ(cell.model().tree.num_segments(), 3u);

    // add passive to all 3 segments in the cell
    for(auto& seg :cell.segments()) {
        seg->add_mechanism(pas_parameters());
    }

    cell.soma()->add_mechanism(hh_parameters());
    cell.segment(2)->add_mechanism(hh_parameters());

    auto& soma_hh = cell.soma()->mechanism("hh");

    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.387);

    cell.segment(1)->set_compartments(4);
    cell.segment(2)->set_compartments(4);

    using fvm_cell = fvm::fvm_cell<double, int>;
    fvm_cell fvcell(cell);
    auto& J = fvcell.jacobian();

    EXPECT_EQ(cell.num_compartments(), 9u);

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

    nest::mc::cell cell;

    // setup global state for the mechanisms
    nest::mc::mechanisms::setup_mechanism_helpers();

    cell.add_soma(12.6157/2.0);
    //auto& props = cell.soma()->properties;

    cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);

    const auto m = cell.model();
    EXPECT_EQ(m.tree.num_segments(), 2u);

    // in this context (i.e. attached to a segment on a high-level cell)
    // a mechanism is essentially a set of parameters
    // - the only "state" is that used to define parameters
    cell.soma()->add_mechanism(hh_parameters());

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

    using fvm_cell = fvm::fvm_cell<double, int>;
    fvm_cell fvcell(cell);
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    fvcell.setup_matrix(0.01);
}

