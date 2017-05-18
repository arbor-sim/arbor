#include <cstddef>
#include <fstream>

#include "../gtest.h"

#include <backends/multicore/fvm.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <util/meta.hpp>
#include <util/rangeutil.hpp>

#include "../test_util.hpp"
#include "../test_common_cells.hpp"

using fvm_cell =
    nest::mc::fvm::fvm_multicell<nest::mc::multicore::backend>;

TEST(fvm_multi, cable)
{
    using namespace nest::mc;

    nest::mc::cell cell=make_cell_ball_and_3stick();

    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), targets, probes);

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
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(cell), targets, probes);

    // This is naughty: removing const from the matrix reference, but is needed
    // to test the build_matrix() method below (which is only accessable
    // through non-const interface).
    //auto& J = const_cast<fvm_cell::matrix_type&>(fvcell.jacobian());
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    // test that the matrix is initialized with sensible values
    //J.build_matrix(0.01);
    fvcell.setup_integration(0.01, 0.01);
    fvcell.step_integration();

    auto& mat = J.state_;
    auto test_nan = [](decltype(mat.u) v) {
        for(auto val : v) if(val != val) return false;
        return true;
    };
    EXPECT_TRUE(test_nan(mat.u(1, J.size())));
    EXPECT_TRUE(test_nan(mat.d));
    EXPECT_TRUE(test_nan(J.solution()));

    // test matrix diagonals for sign
    auto is_pos = [](decltype(mat.u) v) {
        for(auto val : v) if(val<=0.) return false;
        return true;
    };
    auto is_neg = [](decltype(mat.u) v) {
        for(auto val : v) if(val>=0.) return false;
        return true;
    };
    EXPECT_TRUE(is_neg(mat.u(1, J.size())));
    EXPECT_TRUE(is_pos(mat.d));

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
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(cells, targets, probes);

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
    // CV        |   4  |    0
    //
    // The implementation of the stimulus is tested by creating a lowered cell, then
    // testing that the correct currents are injected at the correct control volumes
    // as during the stimulus windows.

    std::vector<fvm_cell::target_handle> targets;
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(singleton_view(cell), targets, probes);

    auto ref = fvcell.find_mechanism("stimulus");
    ASSERT_TRUE(ref) << "no stimuli retrieved from lowered fvm cell: expected 2";

    auto& stims = ref.get();
    EXPECT_EQ(stims->size(), 2u);

    auto I = fvcell.current();

    auto soma_idx = 0u;
    auto dend_idx = 4u;

    // test 1: Test that no current is injected at t=0
    memory::fill(I, 0.);
    stims->set_params(0, 0.1);
    stims->nrn_current();
    for (auto i: I) {
        EXPECT_EQ(i, 0.);
    }

    // test 2: Test that current is injected at soma at t=1
    stims->set_params(1, 0.1);
    stims->nrn_current();
    EXPECT_EQ(I[soma_idx], -0.1);

    // test 3: Test that current is still injected at soma at t=1.5.
    //         Note that we test for injection of -0.2, because the
    //         current contributions are accumulative, and the current
    //         values have not been cleared since the last update.
    stims->set_params(1.5, 0.1);
    stims->nrn_current();
    EXPECT_EQ(I[soma_idx], -0.2);

    // test 4: test at t=10ms, when the the soma stim is not active, and
    //         dendrite stimulus is injecting a current of 0.3 nA
    stims->set_params(10, 0.1);
    stims->nrn_current();
    EXPECT_EQ(I[soma_idx], -0.2);
    EXPECT_EQ(I[dend_idx], -0.3);
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
    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

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
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(util::singleton_view(c), targets, probes);

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

struct handle_info {
    unsigned cell;
    std::string mech;
    unsigned cv;
};

// test handle <-> mechanism/index correspondence
// on a two-cell ball-and-stick system.

void run_target_handle_test(std::vector<handle_info> handles) {
    using namespace nest::mc;

    nest::mc::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_stick()
    };

    EXPECT_EQ(2u, cells[0].num_segments());
    EXPECT_EQ(4u, cells[0].segment(1)->num_compartments());
    EXPECT_EQ(5u, cells[0].num_compartments());

    EXPECT_EQ(2u, cells[1].num_segments());
    EXPECT_EQ(4u, cells[1].segment(1)->num_compartments());
    EXPECT_EQ(5u, cells[1].num_compartments());

    for (auto& x: handles) {
        unsigned seg_id;
        double pos;

        ASSERT_TRUE(x.cell==0 || x.cell==1);
        ASSERT_TRUE(x.cv<5);
        ASSERT_TRUE(x.mech=="expsyn" || x.mech=="exp2syn");

        if (x.cv==0) {
            // place on soma
            seg_id = 0;
            pos = 0;
        }
        else {
            // place on dendrite
            seg_id = 1;
            pos = x.cv/4.0;
        }

        if (x.cell==1) {
            x.cv += 5; // offset for cell 1
        }

        cells[x.cell].add_synapse({seg_id, pos}, parameter_list(x.mech));
    }

    auto n = handles.size();
    std::vector<fvm_cell::target_handle> targets(n);
    std::vector<fvm_cell::probe_handle> probes;

    fvm_cell fvcell;
    fvcell.initialize(cells, targets, probes);

    ASSERT_EQ(n, util::size(targets));
    for (std::size_t i = 0; i<n; ++i) {
        // targets are represented by a pair of mechanism index and instance index
        const auto& mech = fvcell.mechanisms()[targets[i].first];
        const auto& cvidx = mech->node_index();
        EXPECT_EQ(handles[i].mech, mech->name());
        EXPECT_EQ(handles[i].cv, cvidx[targets[i].second]);
    }
}

TEST(fvm_multi, target_handles_onecell)
{
    SCOPED_TRACE("handles: exp2syn only on cell 0");
    std::vector<handle_info> handles0 = {
        {0, "exp2syn",  4},
        {0, "exp2syn",  4},
        {0, "exp2syn",  3},
        {0, "exp2syn",  2},
        {0, "exp2syn",  0},
        {0, "exp2syn",  1},
        {0, "exp2syn",  2}
    };
    run_target_handle_test(handles0);

    SCOPED_TRACE("handles: expsyn only on cell 1");
    std::vector<handle_info> handles1 = {
        {1, "expsyn",  4},
        {1, "expsyn",  4},
        {1, "expsyn",  3},
        {1, "expsyn",  2},
        {1, "expsyn",  0},
        {1, "expsyn",  1},
        {1, "expsyn",  2}
    };
    run_target_handle_test(handles1);
}

TEST(fvm_multi, target_handles_twocell)
{
    SCOPED_TRACE("handles: expsyn only on cells 0 and 1");
    std::vector<handle_info> handles = {
        {0, "expsyn",  4},
        {1, "expsyn",  4},
        {1, "expsyn",  3},
        {0, "expsyn",  2},
        {0, "expsyn",  0},
        {1, "expsyn",  1},
        {1, "expsyn",  2}
    };
    run_target_handle_test(handles);
}

TEST(fvm_multi, target_handles_mixed_synapse)
{
    SCOPED_TRACE("handles: expsyn and exp2syn on cells 0");
    std::vector<handle_info> handles = {
        {0, "expsyn",  4},
        {0, "exp2syn", 4},
        {0, "expsyn",  3},
        {0, "exp2syn", 2},
        {0, "exp2syn", 0},
        {0, "expsyn",  1},
        {0, "expsyn",  2}
    };
    run_target_handle_test(handles);
}

TEST(fvm_multi, target_handles_general)
{
    SCOPED_TRACE("handles: expsyn and exp2syn on cells 0 and 1");
    std::vector<handle_info> handles = {
        {0, "expsyn",  4},
        {1, "exp2syn", 4},
        {1, "expsyn",  3},
        {0, "exp2syn", 2},
        {0, "exp2syn", 0},
        {1, "expsyn",  1},
        {1, "expsyn",  2}
    };
    run_target_handle_test(handles);
}

