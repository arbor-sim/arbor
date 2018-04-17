#include <vector>

#include "../gtest.h"

#include <algorithms.hpp>
#include <backends/fvm_types.hpp>
#include <backends/multicore/fvm.hpp>
#include <backends/multicore/mechanism.hpp>
#include <cell.hpp>
#include <common_types.hpp>
#include <fvm_lowered_cell.hpp>
#include <fvm_lowered_cell_impl.hpp>
#include <load_balance.hpp>
#include <math.hpp>
#include <simulation.hpp>
#include <recipe.hpp>
#include <sampler_map.hpp>
#include <sampling.hpp>
#include <schedule.hpp>
#include <segment.hpp>
#include <util/meta.hpp>
#include <util/maputil.hpp>
#include <util/rangeutil.hpp>

#include "common.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace testing::string_literals;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

// Access to fvm_cell private data:

using shared_state = backend::shared_state;
ACCESS_BIND(std::unique_ptr<shared_state> fvm_cell::*, private_state_ptr, &fvm_cell::state_)

using matrix = arb::matrix<arb::multicore::backend>;
ACCESS_BIND(matrix fvm_cell::*, private_matrix_ptr, &fvm_cell::matrix_)

ACCESS_BIND(std::vector<arb::mechanism_ptr> fvm_cell::*, private_mechanisms_ptr, &fvm_cell::mechanisms_)

arb::mechanism* find_mechanism(fvm_cell& fvcell, const std::string& name) {
    for (auto& mech: fvcell.*private_mechanisms_ptr) {
        if (mech->internal_name()==name) {
            return mech.get();
        }
    }
    return nullptr;
}

// Access to mechanism-internal data:

using mechanism_global_table = std::vector<std::pair<const char*, arb::fvm_value_type*>>;
using mechanism_field_table = std::vector<std::pair<const char*, arb::fvm_value_type**>>;
using mechanism_ion_index_table = std::vector<std::pair<arb::ionKind, backend::iarray*>>;

ACCESS_BIND(\
    mechanism_global_table (arb::multicore::mechanism::*)(),\
    private_global_table_ptr,\
    &arb::multicore::mechanism::global_table)

ACCESS_BIND(\
    mechanism_field_table (arb::multicore::mechanism::*)(),\
    private_field_table_ptr,\
    &arb::multicore::mechanism::field_table)

ACCESS_BIND(\
    mechanism_ion_index_table (arb::multicore::mechanism::*)(),\
    private_ion_index_table_ptr,\
    &arb::multicore::mechanism::ion_index_table)


// TODO: C++14 replace use with generic lambda
struct generic_isnan {
    template <typename V>
    bool operator()(V& v) const { return std::isnan(v); }
} isnan_;

using namespace arb;

TEST(fvm_lowered, matrix_init)
{
    algorithms::generic_is_positive ispos;
    algorithms::generic_is_negative isneg;

    arb::cell cell = make_cell_ball_and_stick();

    ASSERT_EQ(2u, cell.num_segments());
    cell.segment(1)->set_compartments(10);

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(cell), targets, probe_map);

    auto& J = fvcell.*private_matrix_ptr;
    EXPECT_EQ(J.size(), 11u);

    // Test that the matrix is initialized with sensible values

    fvcell.integrate(0.01, 0.01, {}, {});

    auto n = J.size();
    auto& mat = J.state_;

    EXPECT_FALSE(util::any_of(util::subrange_view(mat.u, 1, n), isnan_));
    EXPECT_FALSE(util::any_of(mat.d, isnan_));
    EXPECT_FALSE(util::any_of(J.solution(), isnan_));

    EXPECT_FALSE(util::any_of(util::subrange_view(mat.u, 1, n), ispos));
    EXPECT_FALSE(util::any_of(mat.d, isneg));
}

TEST(fvm_lowered, target_handles) {
    using namespace arb;

    arb::cell cells[] = {
        make_cell_ball_and_stick(),
        make_cell_ball_and_3stick()
    };

    EXPECT_EQ(cells[0].num_segments(), 2u);
    EXPECT_EQ(cells[1].num_segments(), 4u);

    // (in increasing target order)
    cells[0].add_synapse({1, 0.4}, "expsyn");
    cells[0].add_synapse({0, 0.5}, "expsyn");
    cells[1].add_synapse({2, 0.2}, "exp2syn");
    cells[1].add_synapse({2, 0.8}, "expsyn");

    cells[1].add_detector({0, 0}, 3.3);

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0, 1}, cable1d_recipe(cells), targets, probe_map);

    mechanism* expsyn = find_mechanism(fvcell, "expsyn");
    ASSERT_TRUE(expsyn);
    mechanism* exp2syn = find_mechanism(fvcell, "exp2syn");
    ASSERT_TRUE(exp2syn);

    unsigned expsyn_id = expsyn->mechanism_id();
    unsigned exp2syn_id = exp2syn->mechanism_id();

    EXPECT_EQ(4u, targets.size());

    EXPECT_EQ(expsyn_id, targets[0].mech_id);
    EXPECT_EQ(1u, targets[0].mech_index);
    EXPECT_EQ(0u, targets[0].cell_index);

    EXPECT_EQ(expsyn_id, targets[1].mech_id);
    EXPECT_EQ(0u, targets[1].mech_index);
    EXPECT_EQ(0u, targets[1].cell_index);

    EXPECT_EQ(exp2syn_id, targets[2].mech_id);
    EXPECT_EQ(0u, targets[2].mech_index);
    EXPECT_EQ(1u, targets[2].cell_index);

    EXPECT_EQ(expsyn_id, targets[3].mech_id);
    EXPECT_EQ(2u, targets[3].mech_index);
    EXPECT_EQ(1u, targets[3].cell_index);
}

TEST(fvm_lowered, stimulus) {
    // Ball-and-stick with two stimuli:
    //
    //           |stim0 |stim1
    // -----------------------
    // delay     |   5  |    1
    // duration  |  80  |    2
    // amplitude | 0.3  |  0.1
    // CV        |   4  |    0

    std::vector<cell> cells;
    cells.push_back(make_cell_ball_and_stick(false));

    cells[0].add_stimulus({1,1},   {5., 80., 0.3});
    cells[0].add_stimulus({0,0.5}, {1., 2.,  0.1});

    const fvm_size_type soma_cv = 0u;
    const fvm_size_type tip_cv = 4u;

    // now we have two stims :
    //
    //
    // The implementation of the stimulus is tested by creating a lowered cell, then
    // testing that the correct currents are injected at the correct control volumes
    // as during the stimulus windows.

    fvm_discretization D = fvm_discretize(cells);
    const auto& A = D.cv_area;

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(cells), targets, probe_map);

    mechanism* stim = find_mechanism(fvcell, "_builtin_stimulus");
    ASSERT_TRUE(stim);
    EXPECT_EQ(2u, stim->size());

    auto& state = *(fvcell.*private_state_ptr).get();
    auto& J = state.current_density;
    auto& T = state.time;

    // Test that no current is injected at t=0.
    memory::fill(J, 0.);
    memory::fill(T, 0.);
    stim->nrn_current();

    for (auto j: J) {
        EXPECT_EQ(j, 0.);
    }

    // Test that 0.1 nA current is injected at soma at t=1.
    memory::fill(J, 0.);
    memory::fill(T, 1.);
    stim->nrn_current();
    constexpr double unit_factor = 1e-3; // scale A/m²·µm² to nA
    EXPECT_DOUBLE_EQ(-0.1, J[soma_cv]*A[soma_cv]*unit_factor);

    // Test that 0.1 nA is again injected at t=1.5, for a total of 0.2 nA.
    memory::fill(T, 1.);
    stim->nrn_current();
    EXPECT_DOUBLE_EQ(-0.2, J[soma_cv]*A[soma_cv]*unit_factor);

    // Test that at t=10, no more current is injected at soma, and that
    // that 0.3 nA is injected at dendrite tip.
    memory::fill(T, 10.);
    stim->nrn_current();
    EXPECT_DOUBLE_EQ(-0.2, J[soma_cv]*A[soma_cv]*unit_factor);
    EXPECT_DOUBLE_EQ(-0.3, J[tip_cv]*A[tip_cv]*unit_factor);
}

// Test derived mechanism behaviour.

TEST(fvm_lowered, derived_mechs) {
    // Create ball and stick cells with the 'test_kin1' mechanism, which produces
    // a voltage-independent current density of the form a + exp(-t/tau) as a function
    // of time t.
    //
    // 1. Default 'test_kin1': tau = 10 [ms].
    //
    // 2. Specialized version 'custom_kin1' with tau = 20 [ms].
    //
    // 3. Cell with both test_kin1 and custom_kin1.

    std::vector<cell> cells(3);
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
    rec.catalogue().derive("custom_kin1", "test_kin1", {{"tau", 20.0}});

    cell_probe_address where{{1, 0.3}, cell_probe_address::membrane_current};
    rec.add_probe(0, 0, where);
    rec.add_probe(1, 0, where);
    rec.add_probe(2, 0, where);

    {
        // Test initialization and global parameter values.

        std::vector<target_handle> targets;
        probe_association_map<probe_handle> probe_map;

        fvm_cell fvcell;
        fvcell.initialize({0, 1, 2}, rec, targets, probe_map);

        // Both mechanisms will have the same internal name, "test_kin1".

        using fvec = std::vector<fvm_value_type>;
        fvec tau_values;
        for (auto& mech: fvcell.*private_mechanisms_ptr) {
            EXPECT_EQ("test_kin1"_s, mech->internal_name());

            auto cmech = dynamic_cast<multicore::mechanism*>(mech.get());
            ASSERT_TRUE(cmech);

            auto opt_tau_ptr = util::value_by_key((cmech->*private_global_table_ptr)(), "tau"_s);
            ASSERT_TRUE(opt_tau_ptr);
            tau_values.push_back(*opt_tau_ptr.value());
        }
        util::sort(tau_values);
        EXPECT_EQ(fvec({10., 20.}), tau_values);
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
        simulation sim(rec, decomp);
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

// Test area-weighted linear combination of ion species concentrations

TEST(fvm_lowered, weighted_write_ion) {
    // Create a cell with 4 segments (same morphopology as in fvm_layout.ion_weights test):
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
    //
    // The radius of the soma is chosen such that the surface area of soma is
    // the same as a 100µm dendrite, which makes it easier to describe the
    // expected weights.

    cell c;
    c.add_soma(5);

    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 200);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

    for (auto& s: c.segments()) s->set_compartments(1);

    const double con_int = 80;
    const double con_ext = 120;

    // Ca ion reader test_kinlva on CV 1 and 2 via segment 2:
    c.segments()[2] ->add_mechanism("test_kinlva");

    // Ca ion writer test_ca on CV 1 and 3 via segment 3:
    c.segments()[3] ->add_mechanism("test_ca");

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell;
    fvcell.initialize({0}, cable1d_recipe(c), targets, probe_map);

    auto& state = *(fvcell.*private_state_ptr).get();
    auto& ion = state.ion_data.at(ionKind::ca);
    ion.default_int_concentration = con_int;
    ion.default_ext_concentration = con_ext;
    ion.init_concentration();

    std::vector<unsigned> ion_nodes = util::assign_from(ion.node_index_);
    std::vector<unsigned> expected_ion_nodes = {1, 2, 3};
    EXPECT_EQ(expected_ion_nodes, ion_nodes);

    std::vector<double> ion_iconc_weights = util::assign_from(ion.weight_Xi_);
    std::vector<double> expected_ion_iconc_weights = {0.75, 1., 0};
    EXPECT_EQ(expected_ion_iconc_weights, ion_iconc_weights);

    auto test_ca = dynamic_cast<multicore::mechanism*>(find_mechanism(fvcell, "test_ca"));

    auto opt_cai_ptr = util::value_by_key((test_ca->*private_field_table_ptr)(), "cai"_s);
    ASSERT_TRUE(opt_cai_ptr);
    auto& test_ca_cai = *opt_cai_ptr.value();

    auto opt_ca_index_ptr = util::value_by_key((test_ca->*private_ion_index_table_ptr)(), ionKind::ca);
    ASSERT_TRUE(opt_ca_index_ptr);
    auto& test_ca_ca_index = *opt_ca_index_ptr.value();

    double cai_contrib[3] = {200., 0., 300.};
    for (int i = 0; i<2; ++i) {
        test_ca_cai[i] = cai_contrib[test_ca_ca_index[i]];
    }

    std::vector<double> expected_iconc(3);
    for (int i = 0; i<3; ++i) {
        expected_iconc[i] = math::lerp(cai_contrib[i], con_int, ion_iconc_weights[i]);
    }

    ion.init_concentration();
    test_ca->write_ions();
    std::vector<double> ion_iconc = util::assign_from(ion.Xi_);
    EXPECT_EQ(expected_iconc, ion_iconc);
}

