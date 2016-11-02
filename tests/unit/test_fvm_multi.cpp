#include <fstream>

#include "../gtest.h"

#include <common_types.hpp>
#include <cell.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "../test_util.hpp"
#include "../test_common_cells.hpp"

using fvm_cell =
    nest::mc::fvm::fvm_multicell<nest::mc::multicore::fvm_policy>;

TEST(fvm_multi, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell=make_cell_ball_and_3stick();

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

    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), detectors, targets, probes);

    // This is naughty: removing const from the matrix reference, but is needed
    // to test the build_matrix() method below (which is only accessable
    // through non-const interface).
    //auto& J = const_cast<fvm_cell::matrix_type&>(fvcell.jacobian());
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    // test that the matrix is initialized with sensible values
    //J.build_matrix(0.01);
    fvcell.advance(0.01);
    auto test_nan = [](decltype(J.u()) v) {
        for(auto val : v) if(val != val) return false;
        return true;
    };
    EXPECT_TRUE(test_nan(J.u()(1, J.size())));
    EXPECT_TRUE(test_nan(J.d()));
    EXPECT_TRUE(test_nan(J.rhs()));

    // test matrix diagonals for sign
    auto is_pos = [](decltype(J.u()) v) {
        for(auto val : v) if(val<=0.) return false;
        return true;
    };
    auto is_neg = [](decltype(J.u()) v) {
        for(auto val : v) if(val>=0.) return false;
        return true;
    };
    EXPECT_TRUE(is_neg(J.u()(1, J.size())));
    EXPECT_TRUE(is_pos(J.d()));

}

TEST(fvm_multi, multi_init)
{
    using namespace nest::mc;

    nest::mc::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_3stick()
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
}

// test that stimuli are added correctly
TEST(fvm_multi, stimulus)
{
    using namespace nest::mc;
    using util::singleton_view;

    // the default ball and stick has one stimulus at the terminal end of the dendrite
    auto cell = make_cell_ball_and_stick();

    // ... so add a second at the soma to make things more interesting
    cell.add_stimulus({0,0.5}, {1., 2., 0.1});

    // now we have two stims :
    //
    //           |stim0 |stim1
    // -----------------------
    // delay     |   5  |    1
    // duration  |  80  |    2
    // amplitude | 0.3  |  0.1
    // compmnt   |   4  |    0


    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(singleton_view(cell), detectors, targets, probes);

    auto& stim = fvcell.stimuli();
    EXPECT_EQ(stim.size(), 2u);

    EXPECT_EQ(stim[0].first, 4u);
    EXPECT_EQ(stim[1].first, 0u);

    EXPECT_EQ(stim[0].second.delay(), 5.);
    EXPECT_EQ(stim[1].second.delay(), 1.);
    EXPECT_EQ(stim[0].second.duration(), 80.);
    EXPECT_EQ(stim[1].second.duration(),  2.);
    EXPECT_EQ(stim[0].second.amplitude(), 0.3);
    EXPECT_EQ(stim[1].second.amplitude(), 0.1);
}

// test that mechanism indexes are computed correctly
TEST(fvm_multi, mechanism_indexes)
{
    using namespace nest::mc;

    // create a cell with 4 sements:
    // a soma with a branching dendrite
    // - hh on soma and first branch of dendrite (segs 0 and 2)
    // - pas on main dendrite and second branch (segs 1 and 3)
    //
    //              /
    //             pas
    //            /
    // hh---pas--.
    //            \.
    //             hh
    //              \.

    cell c;
    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    c.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);

    auto& segs = c.segments();
    segs[1]->add_mechanism(pas_parameters());
    segs[2]->add_mechanism(hh_parameters());
    segs[3]->add_mechanism(pas_parameters());

    for (auto& seg: segs) {
        if (seg->is_dendrite()) {
            seg->mechanism("membrane").set("r_L", 100);
            seg->set_compartments(4);
        }
    }

    // generate the lowered fvm cell
    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::detector_handle> detectors;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(c), detectors, targets, probes);

    // make vectors with the expected CV indexes for each mechanism
    std::vector<unsigned> hh_index  = {0u, 4u, 5u, 6u, 7u, 8u};
    std::vector<unsigned> pas_index = {0u, 1u, 2u, 3u, 4u, 9u, 10u, 11u, 12u};
    // iterate over mechanisms and test whether they were assigned to the correct CVs
    // TODO : this fails because we do not handle CVs at branching points (including soma) correctly
    for(auto& mech : fvcell.mechanisms()) {
        auto const& n = mech->node_index();
        std::vector<unsigned> ni(n.begin(), n.end());
        if(mech->name()=="hh") {
            EXPECT_EQ(ni, hh_index);
        }
        else if(mech->name()=="pas") {
            EXPECT_EQ(ni, pas_index);
        }
    }

    // similarly, test that the different ion channels were assigned to the correct
    // compartments. In this case, the passive channel has no ion species
    // associated with it, while the hh channel has both pottassium and sodium
    // channels. Hence, we expect sodium and potassium to be present in the same
    // compartments as the hh mechanism.
    {
        auto ni = fvcell.ion_na().node_index();
        std::vector<unsigned> na(ni.begin(), ni.end());
        EXPECT_EQ(na, hh_index);
    }
    {
        auto ni = fvcell.ion_k().node_index();
        std::vector<unsigned> k(ni.begin(), ni.end());
        EXPECT_EQ(k, hh_index);
    }
    {
        // calcium channel should be empty
        EXPECT_EQ(0u, fvcell.ion_ca().node_index().size());
    }
}
