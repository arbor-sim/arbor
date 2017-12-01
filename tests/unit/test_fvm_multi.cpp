#include <vector>

#include "../gtest.h"

#include <backends/multicore/fvm.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <fvm_multicell.hpp>
#include <load_balance.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <sampler_map.hpp>
#include <sampling.hpp>
#include <schedule.hpp>
#include <segment.hpp>
#include <util/meta.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using fvm_cell =
    arb::fvm::fvm_multicell<arb::multicore::backend>;

TEST(fvm_multi, cable)
{
    using namespace arb;

    arb::cell cell=make_cell_ball_and_3stick();

    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(cell), targets, probe_map);

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
    using namespace arb;

    arb::cell cell = make_cell_ball_and_stick();

    const auto m = cell.model();
    EXPECT_EQ(m.tree.num_segments(), 2u);

    auto& soma_hh = (cell.soma()->mechanism("hh")).get();

    soma_hh.set("gnabar", 0.12);
    soma_hh.set("gkbar", 0.036);
    soma_hh.set("gl", 0.0003);
    soma_hh.set("el", -54.3);

    cell.segment(1)->set_compartments(10);

    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(cell), targets, probe_map);

    // This is naughty: removing const from the matrix reference, but is needed
    // to test the build_matrix() method below (which is only accessable
    // through non-const interface).
    //auto& J = const_cast<fvm_cell::matrix_type&>(fvcell.jacobian());
    auto& J = fvcell.jacobian();
    EXPECT_EQ(J.size(), 11u);

    // test that the matrix is initialized with sensible values
    //J.build_matrix(0.01);
    fvcell.setup_integration(0.01, 0.01, {}, {});
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
    using namespace arb;

    arb::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_3stick()
    };

    EXPECT_EQ(cells[0].num_segments(), 2u);
    EXPECT_EQ(cells[0].segment(1)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].num_segments(), 4u);
    EXPECT_EQ(cells[1].segment(1)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].segment(2)->num_compartments(), 4u);
    EXPECT_EQ(cells[1].segment(3)->num_compartments(), 4u);

    cells[0].add_synapse({1, 0.4}, "expsyn");
    cells[0].add_synapse({1, 0.4}, "expsyn");
    cells[1].add_synapse({2, 0.4}, "exp2syn");
    cells[1].add_synapse({3, 0.4}, "expsyn");

    cells[1].add_detector({0, 0}, 3.3);

    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0, 1}, cable1d_recipe(cells), targets, probe_map);

    EXPECT_EQ(4u, targets.size());

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
    using namespace arb;

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
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(cell), targets, probe_map);

    auto ref = fvcell.find_mechanism("stimulus");
    ASSERT_TRUE(ref) << "no stimuli retrieved from lowered fvm cell: expected 2";

    auto& stims = ref.get();
    EXPECT_EQ(stims->size(), 2u);

    auto I = fvcell.current();
    auto A = fvcell.cv_areas();

    auto soma_idx = 0u;
    auto dend_idx = 4u;

    // test 1: Test that no current is injected at t=0
    memory::fill(I, 0.);
    fvcell.set_time_global(0.);
    fvcell.set_time_to_global(0.1);
    stims->set_params();
    stims->nrn_current();
    for (auto i: I) {
        EXPECT_EQ(i, 0.);
    }

    // test 2: Test that current is injected at soma at t=1
    fvcell.set_time_global(1.);
    fvcell.set_time_to_global(1.1);
    stims->nrn_current();
    // take care to convert from A.m^-2 to nA
    EXPECT_EQ(I[soma_idx]/(1e3/A[soma_idx]), -0.1);

    // test 3: Test that current is still injected at soma at t=1.5.
    //         Note that we test for injection of -0.2, because the
    //         current contributions are accumulative, and the current
    //         values have not been cleared since the last update.
    fvcell.set_time_global(1.5);
    fvcell.set_time_to_global(1.6);
    stims->set_params();
    stims->nrn_current();
    EXPECT_EQ(I[soma_idx]/(1e3/A[soma_idx]), -0.2);

    // test 4: test at t=10ms, when the the soma stim is not active, and
    //         dendrite stimulus is injecting a current of 0.3 nA
    fvcell.set_time_global(10.);
    fvcell.set_time_to_global(10.1);
    stims->nrn_current();
    EXPECT_EQ(I[soma_idx]/(1e3/A[soma_idx]), -0.2);
    EXPECT_EQ(I[dend_idx]/(1e3/A[dend_idx]), -0.3);
}

// test that mechanism indexes are computed correctly
TEST(fvm_multi, mechanism_indexes)
{
    using namespace arb;

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
    soma->add_mechanism("hh");

    // add dendrite of length 200 um and diameter 1 um with passive channel
    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

    auto& segs = c.segments();
    segs[1]->add_mechanism("pas");
    segs[2]->add_mechanism("hh");
    segs[3]->add_mechanism("pas");

    for (auto& seg: segs) {
        if (seg->is_dendrite()) {
            seg->rL = 100;
            seg->set_compartments(4);
        }
    }

    // generate the lowered fvm cell
    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(c), targets, probe_map);

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

namespace {
    double wm_impl(double wa, double xa) {
        return wa? xa/wa: 0;
    }

    template <typename... R>
    double wm_impl(double wa, double xa, double w, double x, R... rest) {
        return wm_impl(wa+w, xa+w*x, rest...);
    }

    // Computed weighted mean (w*x + ...) / (w + ...).
    template <typename... R>
    double wmean(double w, double x, R... rest) {
        return wm_impl(w, w*x, rest...);
    }
}

// Test area-weighted linear combination of density mechanism parameters.

TEST(fvm_multi, density_weights) {
    using namespace arb;

    // Create a cell with 4 segments:
    //   - Soma (segment 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - HH mechanism on all segments.
    //   - Dendritic segments are given 3 compartments each.
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/6 of segment 1 and the initial 1/6 of segments 2 and 3.
    //
    // The HH mechanism current density parameters ('gnabar', 'gkbar' and 'gl') are set
    // differently for each segment:
    //
    //   soma:      all default values (gnabar = 0.12, gkbar = .036, gl = .0003)
    //   segment 1: gl = .0002
    //   segment 2: gkbar = .05
    //   segment 3: gkbar = .07, gl = .0004
    //
    // Geometry:
    //   segment 1: 100 µm long, 1 µm diameter cylinder.
    //   segment 2: 200 µm long, diameter linear taper from 1 µm to 0.2 µm.
    //   segment 3: 150 µm long, 0.8 µm diameter cylinder.
    //
    // Use divided compartment view on segments to compute area contributions.

    cell c;
    auto soma = c.add_soma(12.6157/2.0);

    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.1, 200);
    c.add_cable(1, section_kind::dendrite, 0.4, 0.4, 150);

    auto& segs = c.segments();

    double dflt_gkbar = .036;
    double dflt_gnabar = 0.12;
    double dflt_gl = 0.0003;

    double seg1_gl = .0002;
    double seg2_gkbar = .05;
    double seg3_gkbar = .0004;
    double seg3_gl = .0004;

    for (int i = 0; i<4; ++i) {
        segment& seg = *segs[i];
        seg.set_compartments(3);

        mechanism_spec hh("hh");
        switch (i) {
        case 1:
            hh["gl"] = seg1_gl;
            break;
        case 2:
            hh["gkbar"] = seg2_gkbar;
            break;
        case 3:
            hh["gkbar"] = seg3_gkbar;
            hh["gl"] = seg3_gl;
            break;
        default: ;
        }
        seg.add_mechanism(hh);
    }

    int ncv = 10;
    std::vector<double> expected_gkbar(ncv, dflt_gkbar);
    std::vector<double> expected_gnabar(ncv, dflt_gnabar);
    std::vector<double> expected_gl(ncv, dflt_gl);

    double soma_area = soma->area();
    auto seg1_divs = div_compartments<div_compartment_by_ends>(segs[1]->as_cable());
    auto seg2_divs = div_compartments<div_compartment_by_ends>(segs[2]->as_cable());
    auto seg3_divs = div_compartments<div_compartment_by_ends>(segs[3]->as_cable());

    // CV 0: mix of soma and left of segment 1
    expected_gl[0] = wmean(soma_area, dflt_gl, seg1_divs(0).left.area, seg1_gl);

    expected_gl[1] = seg1_gl;
    expected_gl[2] = seg1_gl;

    // CV 3: mix of right of segment 1 and left of segments 2 and 3.
    expected_gkbar[3] = wmean(seg1_divs(2).right.area, dflt_gkbar, seg2_divs(0).left.area, seg2_gkbar, seg3_divs(0).left.area, seg3_gkbar);
    expected_gl[3] = wmean(seg1_divs(2).right.area, seg1_gl, seg2_divs(0).left.area, dflt_gl, seg3_divs(0).left.area, seg3_gl);

    // CV 4-6: just segment 2
    expected_gkbar[4] = seg2_gkbar;
    expected_gkbar[5] = seg2_gkbar;
    expected_gkbar[6] = seg2_gkbar;

    // CV 7-9: just segment 3
    expected_gkbar[7] = seg3_gkbar;
    expected_gkbar[8] = seg3_gkbar;
    expected_gkbar[9] = seg3_gkbar;
    expected_gl[7] = seg3_gl;
    expected_gl[8] = seg3_gl;
    expected_gl[9] = seg3_gl;

    // Generate the lowered fvm cell.
    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(c), targets, probe_map);

    // Check CV area assumptions.
    // Note: area integrator used here and in `fvm_multicell` may differ, and so areas computed may
    // differ some due to rounding area, even given that we're dealing with simple truncated cones
    // for segments. Check relative error within a tolerance of (say) 10 epsilon.
    auto cv_areas = fvcell.cv_areas();
    double area_relerr = 10*std::numeric_limits<double>::epsilon();
    EXPECT_TRUE(testing::near_relative(cv_areas[0],
        soma_area+seg1_divs(0).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(cv_areas[1],
        seg1_divs(0).right.area+seg1_divs(1).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(cv_areas[3],
        seg1_divs(2).right.area+seg2_divs(0).left.area+seg3_divs(0).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(cv_areas[6],
        seg2_divs(2).right.area, area_relerr));

    // Grab the HH parameters from the mechanism.
    EXPECT_EQ(1u, fvcell.mechanisms().size());
    auto& hh_mech = *fvcell.mechanisms().front();

    auto gnabar_field = hh_mech.field_view_ptr("gnabar");
    auto gkbar_field = hh_mech.field_view_ptr("gkbar");
    auto gl_field = hh_mech.field_view_ptr("gl");

    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gnabar, hh_mech.*gnabar_field));
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gkbar, hh_mech.*gkbar_field));
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gl, hh_mech.*gl_field));
}

// Test specialized mechanism behaviour.

TEST(fvm_multi, specialized_mechs) {
    using namespace arb;

    // Create ball and stick cells with the 'test_kin1' mechanism, which produces
    // a voltage-independent current density of the form a + exp(-t/tau) as a function
    // of time t.
    //
    // 1. Default 'test_kin1': tau = 10 [ms].
    //
    // 2. Specialized version 'custom_kin1' with tau = 20 [ms].
    //
    // 3. Cell with both test_kin1 and custom_kin1.

    specialized_mechanism custom_kin1 = {"test_kin1", {{"tau", 20.0}}};

    cell cells[3];

    for (int i = 0; i<3; ++i) {
        cell& c = cells[i];
        c.add_soma(6.0);
        c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);

        c.segment(1)->set_compartments(4);
        for (auto& seg: c.segments()) {
            if (!seg->is_soma()) {
                seg->as_cable()->set_compartments(4);
            }

            switch (i) {
            case 0:
                seg->add_mechanism("test_kin1");
                break;
            case 1:
                seg->add_mechanism("custom_kin1");
                break;
            case 2:
                seg->add_mechanism("test_kin1");
                seg->add_mechanism("custom_kin1");
                break;
            }
        }
    }

    cable1d_recipe rec(cells);
    rec.add_specialized_mechanism("custom_kin1", custom_kin1);

    cell_probe_address where{{1, 0.3}, cell_probe_address::membrane_current};
    rec.add_probe(0, 0, where);
    rec.add_probe(1, 0, where);
    rec.add_probe(2, 0, where);

    {
        // Test initialization and global parameter values.

        std::vector<fvm_cell::target_handle> targets;
        probe_association_map<fvm_cell::probe_handle> probe_map;

        fvm_cell fvcell;
        fvcell.initialize({0, 1, 2}, rec, targets, probe_map);

        std::map<std::string, fvm_cell::mechanism*> mechmap;
        for (auto& m: fvcell.mechanisms()) {
            // (names of mechanisms should _all_ be 'test_kin1', but aliases will differ)
            EXPECT_EQ("test_kin1", m->name());
            mechmap[m->alias()] = m.get();
        }

        ASSERT_EQ(2u, mechmap.size());
        EXPECT_NE(0u, mechmap.count("test_kin1"));
        EXPECT_NE(0u, mechmap.count("custom_kin1"));

        // Both mechanisms are of the same type, so we can use the
        // same member pointer.
        auto fptr = mechmap.begin()->second->field_value_ptr("tau");

        ASSERT_NE(nullptr, fptr);

        EXPECT_EQ(10.0, mechmap["test_kin1"]->*fptr);
        EXPECT_EQ(20.0, mechmap["custom_kin1"]->*fptr);
    }

    {
        // Test dynamics:
        // 1. Current at same point on cell 0 at time 10 ms should equal that
        //    on cell 1 at time 20 ms.
        // 2. Current for cell 2 should be sum of currents for cells 0 and 1 at any given time.

        std::vector<double> samples[3];

        sampler_function sampler = [&](cell_member_type pid, probe_tag, std::size_t n, const sample_record* records) {
            for (std::size_t i = 0; i<n; ++i) {
                double v = *util::any_cast<const double*>(records[i].data);
                samples[pid.gid].push_back(v);
            }
        };

        float times[] = {10.f, 20.f};

        auto decomp = partition_load_balance(rec, hw::node_info{1u, 0u});
        model sim(rec, decomp);
        sim.add_sampler(all_probes, explicit_schedule(times), sampler);
        sim.run(30.0, 1.f/1024);

        ASSERT_EQ(2u, samples[0].size());
        ASSERT_EQ(2u, samples[1].size());
        ASSERT_EQ(2u, samples[2].size());

        // Integration isn't exact: let's aim for one part in 10'000.
        double relerr = 0.0001;
        EXPECT_TRUE(testing::near_relative(samples[0][0], samples[1][1], relerr));
        EXPECT_TRUE(testing::near_relative(samples[0][0]+samples[1][0], samples[2][0], relerr));
        EXPECT_TRUE(testing::near_relative(samples[0][1]+samples[1][1], samples[2][1], relerr));
    }
}

// Test synapses with differing parameter settings.

TEST(fvm_multi, synapse_parameters) {
    using namespace arb;

    cell c;
    c.add_soma(6.0);
    c.add_cable(0, section_kind::dendrite, 0.4, 0.4, 100.0);
    c.segment(1)->set_compartments(4);

    // Add synapses out-of-order, with a parameter value a function of position
    // on the segment, so we can test that parameters are properly associated
    // after re-ordering.

    struct pset {
        double x; // segment position
        double tau1;
        double tau2;
    };

    pset settings[] = {
        {0.8, 1.5, 2.5},
        {0.1, 1.6, 3.7},
        {0.5, 1.7, 3.6},
        {0.6, 0.8, 2.5},
        {0.4, 0.9, 3.4},
        {0.9, 1.1, 2.3}
    };

    for (auto s: settings) {
        mechanism_spec m("exp2syn");
        c.add_synapse({1, s.x}, m.set("tau1", s.tau1).set("tau2", s.tau2));
    }

    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(c), targets, probe_map);

    EXPECT_EQ(1u, fvcell.mechanisms().size());
    auto& exp2syn_mech = *fvcell.mechanisms().front();

    auto tau1_ptr = exp2syn_mech.field_view_ptr("tau1");
    auto tau2_ptr = exp2syn_mech.field_view_ptr("tau2");

    // Compare tau1, tau2 values from settings and from mechanism, ignoring order.
    std::set<std::pair<double, double>> expected;
    for (auto s: settings) {
        expected.insert({s.tau1, s.tau2});
    }

    unsigned n = exp2syn_mech.size();
    ASSERT_EQ(util::size(settings), n);

    std::set<std::pair<double, double>> values;
    for (unsigned i = 0; i<n; ++i) {
        values.insert({(exp2syn_mech.*tau1_ptr)[i], (exp2syn_mech.*tau2_ptr)[i]});
    }

    EXPECT_EQ(expected, values);
}

struct handle_info {
    unsigned cell;
    std::string mech;
    unsigned cv;
};

// test handle <-> mechanism/index correspondence
// on a two-cell ball-and-stick system.

void run_target_handle_test(std::vector<handle_info> all_handles) {
    using namespace arb;

    arb::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_stick()
    };

    EXPECT_EQ(2u, cells[0].num_segments());
    EXPECT_EQ(4u, cells[0].segment(1)->num_compartments());
    EXPECT_EQ(5u, cells[0].num_compartments());

    EXPECT_EQ(2u, cells[1].num_segments());
    EXPECT_EQ(4u, cells[1].segment(1)->num_compartments());
    EXPECT_EQ(5u, cells[1].num_compartments());

    std::vector<std::vector<handle_info>> handles(2);

    for (auto x: all_handles) {
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

        cells[x.cell].add_synapse({seg_id, pos}, x.mech);
        handles[x.cell].push_back(x);
    }

    auto n = all_handles.size();
    std::vector<fvm_cell::target_handle> targets;
    probe_association_map<fvm_cell::probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0, 1}, cable1d_recipe(cells), targets, probe_map);

    ASSERT_EQ(n, util::size(targets));
    unsigned i = 0;
    for (unsigned ci = 0; ci<=1; ++ci) {
        for (auto h: handles[ci]) {
            // targets are represented by a pair of mechanism index and instance index
            const auto& mech = fvcell.mechanisms()[targets[i].mech_id];
            const auto& cvidx = mech->node_index();
            EXPECT_EQ(h.mech, mech->name());
            EXPECT_EQ(h.cv, cvidx[targets[i].mech_index]);
            EXPECT_EQ(h.cell, targets[i].cell_index);
            ++i;
        }
    }
}

TEST(fvm_multi, target_handles_onecell)
{
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
    }

    {
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
}

TEST(fvm_multi, target_handles_twocell)
{
    SCOPED_TRACE("handles: expsyn only on cells 0 and 1");
    std::vector<handle_info> handles = {
        {0, "expsyn",  0},
        {1, "expsyn",  3},
        {0, "expsyn",  2},
        {1, "expsyn",  2},
        {0, "expsyn",  4},
        {1, "expsyn",  1},
        {1, "expsyn",  4}
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
        {0, "exp2syn", 2},
        {0, "exp2syn", 0},
        {1, "exp2syn", 4},
        {1, "expsyn",  3},
        {1, "expsyn",  1},
        {1, "expsyn",  2}
    };
    run_target_handle_test(handles);
}

// Test area-weighted linear combination of ion species concentrations

TEST(fvm_multi, ion_weights) {
    using namespace arb;

    // Create a cell with 4 segments:
    //   - Soma (segment 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - Dendritic segments are given 1 compartments each.
    //
    //         /
    //        d2
    //       /
    //   s0-d1
    //       \.
    //        d3
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/2 of segment 1 and the initial 1/2 of segments 2 and 3.
    //
    // Geometry:
    //   soma 0: radius 5 µm
    //   dend 1: 100 µm long, 1 µm diameter cynlinder
    //   dend 2: 200 µm long, 1 µm diameter cynlinder
    //   dend 3: 100 µm long, 1 µm diameter cynlinder
    // The radius of the soma is chosen such that the surface area of soma is
    // the same as a 100µm dendrite, which makes it easier to describe the
    // expected weights.

    cell c;
    c.add_soma(5);

    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 200);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

    for (auto& s: c.segments()) s->set_compartments(1);

    std::vector<std::vector<int>> seg_sets = {
        {0}, {0,2}, {2, 3}, {0, 1, 2, 3},
    };
    std::vector<std::vector<unsigned>> expected_nodes = {
        {0}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3},
    };
    std::vector<std::vector<fvm_value_type>> expected_wght = {
        {1./3}, {1./3, 1./2, 0.}, {1./3, 1./4, 0., 0.}, {0., 0., 0., 0.},
    };

    double con_int = 80;
    double con_ext = 120;
    for (auto run=0u; run<seg_sets.size(); ++run) {
        for (auto i: seg_sets[run]) {
            c.segments()[i]->add_mechanism(mechanism_spec("test_ca"));
        }

        std::vector<fvm_cell::target_handle> targets;
        probe_association_map<fvm_cell::probe_handle> probe_map;

        fvm_cell fvcell;
        fvcell.initialize({0}, cable1d_recipe(c), targets, probe_map);

        auto& ion = fvcell.ion_ca();
        ion.default_int_concentration = con_int;
        ion.default_ext_concentration = con_ext;
        ion.init_concentration();

        auto& nodes = expected_nodes[run];
        auto& weights = expected_wght[run];
        auto ncv = nodes.size();
        EXPECT_EQ(ncv, ion.node_index().size());
        for (auto i: util::make_span(0, ncv)) {
            EXPECT_EQ(nodes[i], ion.node_index()[i]);
            EXPECT_FLOAT_EQ(weights[i], ion.internal_concentration_weights()[i]);

            EXPECT_EQ(con_ext, ion.external_concentration()[i]);
            EXPECT_FLOAT_EQ(1.0, ion.external_concentration_weights()[i]);
        }
    }
}
