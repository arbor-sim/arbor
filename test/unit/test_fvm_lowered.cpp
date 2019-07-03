#include <string>
#include <vector>

#include "../gtest.h"

#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/math.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/segment.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>
#include <arbor/schedule.hpp>

#include "algorithms.hpp"
#include "backends/multicore/fvm.hpp"
#include "backends/multicore/mechanism.hpp"
#include "execution_context.hpp"
#include "fvm_lowered_cell.hpp"
#include "fvm_lowered_cell_impl.hpp"
#include "sampler_map.hpp"
#include "util/meta.hpp"
#include "util/maputil.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "unit_test_catalogue.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace std::string_literals;

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

arb::mechanism* find_mechanism(fvm_cell& fvcell, int index) {
    auto& mechs = fvcell.*private_mechanisms_ptr;
    return index<(int)mechs.size()? mechs[index].get(): nullptr;
}

// Access to mechanism-internal data:

using mechanism_global_table = std::vector<std::pair<const char*, arb::fvm_value_type*>>;
using mechanism_field_table = std::vector<std::pair<const char*, arb::fvm_value_type**>>;
using mechanism_ion_index_table = std::vector<std::pair<const char*, backend::iarray*>>;

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

using namespace arb;

class gap_recipe_0: public recipe {
public:
    gap_recipe_0() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;
        c.add_soma(20);
        c.add_gap_junction({0, 1});
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }
    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
        switch (gid) {
            case 0 :
                return {gap_junction_connection({5, 0}, {0, 0}, 0.1)};
            case 2 :
                return {
                        gap_junction_connection({3, 0}, {2, 0}, 0.1),
                };
            case 3 :
                return {
                        gap_junction_connection({7, 0}, {3, 0}, 0.1),
                        gap_junction_connection({3, 0}, {2, 0}, 0.1)
                };
            case 5 :
                return {gap_junction_connection({5, 0}, {0, 0}, 0.1)};
            case 7 :
                return {
                        gap_junction_connection({3, 0}, {7, 0}, 0.1),
                };
            default :
                return {};
        }
    }

private:
    cell_size_type size_ = 12;
};

class gap_recipe_1: public recipe {
public:
    gap_recipe_1() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        cable_cell c;
        c.add_soma(20);
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }

private:
    cell_size_type size_ = 12;
};

class gap_recipe_2: public recipe {
public:
    gap_recipe_2() {}

    cell_size_type num_cells() const override {
        return size_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type) const override {
        cable_cell c;
        c.add_soma(20);
        c.add_gap_junction({0,1});
        return {std::move(c)};
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }
    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
        switch (gid) {
            case 0 :
                return {
                        gap_junction_connection({2, 0}, {0, 0}, 0.1),
                        gap_junction_connection({3, 0}, {0, 0}, 0.1),
                        gap_junction_connection({5, 0}, {0, 0}, 0.1)
                };
            case 2 :
                return {
                        gap_junction_connection({0, 0}, {2, 0}, 0.1),
                        gap_junction_connection({3, 0}, {2, 0}, 0.1),
                        gap_junction_connection({5, 0}, {2, 0}, 0.1)
                };
            case 3 :
                return {
                        gap_junction_connection({0, 0}, {3, 0}, 0.1),
                        gap_junction_connection({2, 0}, {3, 0}, 0.1),
                        gap_junction_connection({5, 0}, {3, 0}, 0.1)
                };
            case 5 :
                return {
                        gap_junction_connection({2, 0}, {5, 0}, 0.1),
                        gap_junction_connection({3, 0}, {5, 0}, 0.1),
                        gap_junction_connection({0, 0}, {5, 0}, 0.1)
                };
            default :
                return {};
        }
    }

private:
    cell_size_type size_ = 12;
};


TEST(fvm_lowered, matrix_init)
{
    execution_context context;

    auto isnan = [](auto v) { return std::isnan(v); };
    auto ispos = [](auto v) { return v>0; };
    auto isneg = [](auto v) { return v<0; };

    cable_cell cell = make_cell_ball_and_stick();

    ASSERT_EQ(2u, cell.num_segments());
    cell.segment(1)->set_compartments(10);

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell(context);
    fvcell.initialize({0}, cable1d_recipe(cell), cell_to_intdom, targets, probe_map);

    auto& J = fvcell.*private_matrix_ptr;
    EXPECT_EQ(J.size(), 11u);

    // Test that the matrix is initialized with sensible values

    fvcell.integrate(0.01, 0.01, {}, {});

    auto n = J.size();
    auto& mat = J.state_;

    EXPECT_FALSE(util::any_of(util::subrange_view(mat.u, 1, n), isnan));
    EXPECT_FALSE(util::any_of(mat.d, isnan));
    EXPECT_FALSE(util::any_of(J.solution(), isnan));

    EXPECT_FALSE(util::any_of(util::subrange_view(mat.u, 1, n), ispos));
    EXPECT_FALSE(util::any_of(mat.d, isneg));
}

TEST(fvm_lowered, target_handles) {
    using namespace arb;

    execution_context context;

    cable_cell cells[] = {
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
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    auto test_target_handles = [&](fvm_cell& cell) {
        mechanism *expsyn = find_mechanism(cell, "expsyn");
        ASSERT_TRUE(expsyn);
        mechanism *exp2syn = find_mechanism(cell, "exp2syn");
        ASSERT_TRUE(exp2syn);

        unsigned expsyn_id = expsyn->mechanism_id();
        unsigned exp2syn_id = exp2syn->mechanism_id();

        EXPECT_EQ(4u, targets.size());

        EXPECT_EQ(expsyn_id, targets[0].mech_id);
        EXPECT_EQ(1u, targets[0].mech_index);
        EXPECT_EQ(0u, targets[0].intdom_index);

        EXPECT_EQ(expsyn_id, targets[1].mech_id);
        EXPECT_EQ(0u, targets[1].mech_index);
        EXPECT_EQ(0u, targets[1].intdom_index);

        EXPECT_EQ(exp2syn_id, targets[2].mech_id);
        EXPECT_EQ(0u, targets[2].mech_index);
        EXPECT_EQ(1u, targets[2].intdom_index);

        EXPECT_EQ(expsyn_id, targets[3].mech_id);
        EXPECT_EQ(2u, targets[3].mech_index);
        EXPECT_EQ(1u, targets[3].intdom_index);
    };

    fvm_cell fvcell0(context);
    fvcell0.initialize({0, 1}, cable1d_recipe(cells, true), cell_to_intdom, targets, probe_map);
    test_target_handles(fvcell0);

    fvm_cell fvcell1(context);
    fvcell1.initialize({0, 1}, cable1d_recipe(cells, false), cell_to_intdom, targets, probe_map);
    test_target_handles(fvcell1);

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

    execution_context context;

    std::vector<cable_cell> cells;
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
    std::vector<fvm_index_type> cell_to_intdom(cells.size(), 0);

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    fvm_discretization D = fvm_discretize(cells, gprop.default_parameters);
    const auto& A = D.cv_area;

    std::vector<target_handle> targets;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell(context);
    fvcell.initialize({0}, cable1d_recipe(cells), cell_to_intdom, targets, probe_map);

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

    std::vector<cable_cell> cells(3);
    for (int i = 0; i<3; ++i) {
        cable_cell& c = cells[i];
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
        std::vector<fvm_index_type> cell_to_intdom;
        probe_association_map<probe_handle> probe_map;

        execution_context context;
        fvm_cell fvcell(context);
        fvcell.initialize({0, 1, 2}, rec, cell_to_intdom, targets, probe_map);

        // Both mechanisms will have the same internal name, "test_kin1".

        using fvec = std::vector<fvm_value_type>;
        fvec tau_values;
        for (auto& mech: fvcell.*private_mechanisms_ptr) {
            EXPECT_EQ("test_kin1"s, mech->internal_name());

            auto cmech = dynamic_cast<multicore::mechanism*>(mech.get());
            ASSERT_TRUE(cmech);

            auto opt_tau_ptr = util::value_by_key((cmech->*private_global_table_ptr)(), "tau"s);
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

        auto ctx = make_context();
        auto decomp = partition_load_balance(rec, ctx);
        simulation sim(rec, decomp, ctx);
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

// Test that ion charge is propagated into mechanism variable.

TEST(fvm_lowered, read_valence) {
    execution_context context;

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    {
        std::vector<cable_cell> cells(1);

        cable_cell& c = cells[0];
        auto soma = c.add_soma(6.0);
        soma->add_mechanism("test_ca_read_valence");

        cable1d_recipe rec(cells);
        rec.catalogue() = make_unit_test_catalogue();

        fvm_cell fvcell(context);
        fvcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

        // test_ca_read_valence initialization should write ca ion valence
        // to state variable 'record_zca':

        auto mech_ptr = dynamic_cast<multicore::mechanism*>(find_mechanism(fvcell, "test_ca_read_valence"));
        auto opt_record_z_ptr = util::value_by_key((mech_ptr->*private_field_table_ptr)(), "record_z"s);

        ASSERT_TRUE(opt_record_z_ptr);
        auto& record_z = *opt_record_z_ptr.value();
        ASSERT_EQ(2.0, record_z[0]);
    }

    {
        // Check ion renaming.
        std::vector<cable_cell> cells(1);

        cable_cell& c = cells[0];
        auto soma = c.add_soma(6.0);
        soma->add_mechanism("cr_read_valence");

        cable1d_recipe rec(cells);
        rec.catalogue() = make_unit_test_catalogue();

        rec.catalogue().derive("na_read_valence", "test_ca_read_valence", {}, {{"ca", "na"}});
        rec.catalogue().derive("cr_read_valence", "na_read_valence", {}, {{"na", "mn"}});
        rec.add_ion("mn", 7, 0, 0, 0);

        fvm_cell fvcell(context);
        fvcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

        auto cr_mech_ptr = dynamic_cast<multicore::mechanism*>(find_mechanism(fvcell, 0));
        auto cr_opt_record_z_ptr = util::value_by_key((cr_mech_ptr->*private_field_table_ptr)(), "record_z"s);

        ASSERT_TRUE(cr_opt_record_z_ptr);
        auto& cr_record_z = *cr_opt_record_z_ptr.value();
        ASSERT_EQ(7.0, cr_record_z[0]);
    }
}

// Test correct scaling of ionic currents in reading and writing

TEST(fvm_lowered, ionic_currents) {
    cable_cell c;
    auto soma = c.add_soma(6.0);

    // Mechanism parameter is in NMODL units, i.e. mA/cm².

    const double jca = 1.5;
    soma->add_mechanism(mechanism_desc("fixed_ica_current").set("ica_density", jca));

    // Mechanism models a well-mixed fixed-depth volume without replenishment,
    // giving a linear response to ica over time.
    //
    //     cai' = - coeff · ica
    //
    // with NMODL units: cai' [mM/ms]; ica [mA/cm²], giving coeff in [mol/cm/C].

    const double coeff = 0.5;
    soma->add_mechanism(mechanism_desc("linear_ca_conc").set("coeff", coeff));

    cable1d_recipe rec(c);
    rec.catalogue() = make_unit_test_catalogue();

    execution_context context;

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell(context);
    fvcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    auto& state = *(fvcell.*private_state_ptr).get();
    auto& ion = state.ion_data.at("ca"s);

    // Ionic current should be 15 A/m², and initial concentration zero.
    EXPECT_EQ(15, ion.iX_[0]);
    EXPECT_EQ(0, ion.Xi_[0]);

    // Integration should be (effectively) exact, so check for linear response.
    const double time = 12; // [ms]
    (void)fvcell.integrate(time, 0.1, {}, {});
    double expected_Xi = -time*coeff*jca;
    EXPECT_NEAR(expected_Xi, ion.Xi_[0], 1e-6);
}

// Test correct scaling of an ionic current updated via a point mechanism

TEST(fvm_lowered, point_ionic_current) {
    cable_cell c;

    double r = 6.0; // [µm]
    c.add_soma(6.0);

    double soma_area_m2 = 4*math::pi<double>*r*r*1e-12; // [m²]

    // Event weight is translated by point_ica_current into a current contribution in nA.
    c.add_synapse({0u, 0.5}, "point_ica_current");

    cable1d_recipe rec(c);
    rec.catalogue() = make_unit_test_catalogue();

    execution_context context;

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell(context);
    fvcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    // Only one target, corresponding to our point process on soma.
    double ica_nA = 12.3;
    deliverable_event ev = {0.04, target_handle{0, 0, 0}, (float)ica_nA};

    auto& state = *(fvcell.*private_state_ptr).get();
    auto& ion = state.ion_data.at("ca"s);

    // Ionic current should be 0 A/m² after initialization.
    EXPECT_EQ(0, ion.iX_[0]);

    // Ionic current should be ica_nA/soma_area after integrating past event time.
    const double time = 0.5; // [ms]
    (void)fvcell.integrate(time, 0.01, {ev}, {});

    double expected_iX = ica_nA*1e-9/soma_area_m2;
    EXPECT_FLOAT_EQ(expected_iX, ion.iX_[0]);
}

// Test area-weighted linear combination of ion species concentrations

TEST(fvm_lowered, weighted_write_ion) {
    // Create a cell with 4 segments (same morphopology as in fvm_layout.ion_weights test):
    //   - Soma (segment 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - Dendritic segments are given 1 compartments each.
    //
    //          /
    //         d2
    //        /
    //   s0-d1
    //        \.
    //         d3
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/2 of segment 1 and the initial 1/2 of segments 2 and 3.
    //
    // Geometry:
    //   soma 0: radius 5 µm
    //   dend 1: 100 µm long, 1 µm diameter cylinder
    //   dend 2: 200 µm long, 1 µm diameter cylinder
    //   dend 3: 100 µm long, 1 µm diameter cylinder
    //
    // The radius of the soma is chosen such that the surface area of soma is
    // the same as a 100 µm dendrite, which makes it easier to describe the
    // expected weights.

    execution_context context;

    cable_cell c;
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

    cable1d_recipe rec(c);
    rec.add_ion("ca", 2, con_int, con_ext, 0.0);

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell fvcell(context);
    fvcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    auto& state = *(fvcell.*private_state_ptr).get();
    auto& ion = state.ion_data.at("ca"s);
    ion.init_concentration();

    std::vector<unsigned> ion_nodes = util::assign_from(ion.node_index_);
    std::vector<unsigned> expected_ion_nodes = {1, 2, 3};
    EXPECT_EQ(expected_ion_nodes, ion_nodes);

    std::vector<double> ion_init_iconc = util::assign_from(ion.init_Xi_);
    std::vector<double> expected_init_iconc = {0.75*con_int, 1.*con_int, 0};
    EXPECT_EQ(expected_init_iconc, ion_init_iconc);

    auto test_ca = dynamic_cast<multicore::mechanism*>(find_mechanism(fvcell, "test_ca"));

    auto opt_cai_ptr = util::value_by_key((test_ca->*private_field_table_ptr)(), "cai"s);
    ASSERT_TRUE(opt_cai_ptr);
    auto& test_ca_cai = *opt_cai_ptr.value();

    auto opt_ca_index_ptr = util::value_by_key((test_ca->*private_ion_index_table_ptr)(), "ca"s);
    ASSERT_TRUE(opt_ca_index_ptr);
    auto& test_ca_ca_index = *opt_ca_index_ptr.value();

    double cai_contrib[3] = {200., 0., 300.};
    double test_ca_weight[3] = {0.25, 0., 1.};

    for (int i = 0; i<2; ++i) {
        test_ca_cai[i] = cai_contrib[test_ca_ca_index[i]];
    }

    std::vector<double> expected_iconc(3);
    for (int i = 0; i<3; ++i) {
        expected_iconc[i] = test_ca_weight[i]*cai_contrib[i] + ion_init_iconc[i];
    }

    ion.init_concentration();
    test_ca->write_ions();
    std::vector<double> ion_iconc = util::assign_from(ion.Xi_);
    EXPECT_EQ(expected_iconc, ion_iconc);
}

TEST(fvm_lowered, gj_coords_simple) {
    using pair = std::pair<int, int>;

    class gap_recipe: public recipe {
    public:
        gap_recipe() {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return {};
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            conns.push_back(gap_junction_connection({(gid+1)%2, 0}, {gid, 0}, 0.5));
            return conns;
        }

    protected:
        cell_size_type n_ = 2;
    };

    execution_context context;
    fvm_cell fvcell(context);

    gap_recipe rec;
    std::vector<cable_cell> cells;
    cable_cell c, d;
    c.add_soma(2.1);
    c.add_cable(0, section_kind::dendrite, 0.3, 0.2, 10);
    c.segment(1)->set_compartments(5);
    c.add_gap_junction({1, 0.8});
    cells.push_back(std::move(c));

    d.add_soma(2.4);
    d.add_cable(0, section_kind::dendrite, 0.3, 0.2, 10);
    d.segment(1)->set_compartments(2);
    d.add_gap_junction({1, 1});
    cells.push_back(std::move(d));

    fvm_discretization D = fvm_discretize(cells, neuron_parameter_defaults);

    std::vector<cell_gid_type> gids = {0, 1};
    auto GJ = fvcell.fvm_gap_junctions(cells, gids, rec, D);

    auto weight = [&](fvm_value_type g, fvm_index_type i){
        return g * 1e3 / D.cv_area[i];
    };

    EXPECT_EQ(pair({4,8}), GJ[0].loc);
    EXPECT_EQ(weight(0.5, 4), GJ[0].weight);

    EXPECT_EQ(pair({8,4}), GJ[1].loc);
    EXPECT_EQ(weight(0.5, 8), GJ[1].weight);
}

TEST(fvm_lowered, gj_coords_complex) {
    using pair = std::pair<int, int>;

    class gap_recipe: public recipe {
    public:
        gap_recipe() {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return {};
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            switch (gid) {
                case 0 :  return {
                            gap_junction_connection({2, 0}, {0, 1}, 0.01),
                            gap_junction_connection({1, 0}, {0, 0}, 0.03),
                            gap_junction_connection({1, 1}, {0, 0}, 0.04)
                    };
                case 1 :  return {
                            gap_junction_connection({0, 0}, {1, 0}, 0.03),
                            gap_junction_connection({0, 0}, {1, 1}, 0.04),
                            gap_junction_connection({2, 1}, {1, 2}, 0.02),
                            gap_junction_connection({2, 2}, {1, 3}, 0.01)
                    };
                case 2 :  return {
                            gap_junction_connection({0, 1}, {2, 0}, 0.01),
                            gap_junction_connection({1, 2}, {2, 1}, 0.02),
                            gap_junction_connection({1, 3}, {2, 2}, 0.01)
                    };
                default : return {};
            }
            return conns;
        }

    protected:
        cell_size_type n_ = 3;
    };

    execution_context context;
    fvm_cell fvcell(context);

    gap_recipe rec;
    cable_cell c0, c1, c2;
    std::vector<cable_cell> cells;

    // Make 3 cells
    c0.add_soma(2.1);
    c0.add_cable(0, section_kind::dendrite, 0.3, 0.2, 8);
    c0.segment(1)->set_compartments(4);

    c1.add_soma(1.4);
    c1.add_cable(0, section_kind::dendrite, 0.3, 0.5, 12);
    c1.segment(1)->set_compartments(6);
    c1.add_cable(1, section_kind::dendrite, 0.3, 0.2, 9);
    c1.segment(2)->set_compartments(3);
    c1.add_cable(1, section_kind::dendrite, 0.2, 0.2, 15);
    c1.segment(3)->set_compartments(5);

    c2.add_soma(2.9);
    c2.add_cable(0, section_kind::dendrite, 0.3, 0.5, 4);
    c2.segment(1)->set_compartments(2);
    c2.add_cable(1, section_kind::dendrite, 0.4, 0.2, 6);
    c2.segment(2)->set_compartments(2);
    c2.add_cable(1, section_kind::dendrite, 0.1, 0.2, 8);
    c2.segment(3)->set_compartments(2);
    c2.add_cable(2, section_kind::dendrite, 0.2, 0.2, 4);
    c2.segment(4)->set_compartments(2);
    c2.add_cable(2, section_kind::dendrite, 0.2, 0.2, 4);
    c2.segment(5)->set_compartments(2);

    // Add 5 gap junctions
    c0.add_gap_junction({1, 1});
    c0.add_gap_junction({1, 0.5});

    c1.add_gap_junction({2, 1});
    c1.add_gap_junction({1, 1});
    c1.add_gap_junction({1, 0.45});
    c1.add_gap_junction({1, 0.1});

    c2.add_gap_junction({1, 0.5});
    c2.add_gap_junction({4, 1});
    c2.add_gap_junction({2, 1});

    cells.push_back(std::move(c0));
    cells.push_back(std::move(c1));
    cells.push_back(std::move(c2));

    std::vector<fvm_index_type> cell_to_intdom;

    std::vector<cell_gid_type> gids = {0, 1, 2};

    fvcell.fvm_intdom(rec, gids, cell_to_intdom);
    fvm_discretization D = fvm_discretize(cells, neuron_parameter_defaults);

    auto GJ = fvcell.fvm_gap_junctions(cells, gids, rec, D);
    EXPECT_EQ(10u, GJ.size());

    auto weight = [&](fvm_value_type g, fvm_index_type i){
        return g * 1e3 / D.cv_area[i];
    };

    std::vector<pair> expected_loc = {{4, 14}, {4,11}, {2,21}, {14, 4}, {11,4} ,{8,28}, {6, 24}, {21,2}, {28,8}, {24, 6}};
    std::vector<double> expected_weight = {
            weight(0.03, 4), weight(0.04, 4), weight(0.01, 2), weight(0.03, 14), weight(0.04, 11),
            weight(0.02, 8), weight(0.01, 6), weight(0.01, 21), weight(0.02, 28), weight(0.01, 24)
    };

    for (unsigned i = 0; i < GJ.size(); i++) {
        bool found = false;
        for (unsigned j = 0; j < expected_loc.size(); j++) {
            if (expected_loc[j].first ==  GJ[i].loc.first && expected_loc[j].second ==  GJ[i].loc.second) {
                found = true;
                EXPECT_EQ(expected_weight[j], GJ[i].weight);
                break;
            }
        }
        EXPECT_TRUE(found);
    }
    std::cout << std::endl;
}

TEST(fvm_lowered, cell_group_gj) {
    using pair = std::pair<int, int>;

    class gap_recipe: public recipe {
    public:
        gap_recipe() {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return {};
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            if (gid % 2 == 0) {
                // connect 5 of the first 10 cells in a ring; connect 5 of the second 10 cells in a ring
                auto next_cell = gid == 8 ? 0 : (gid == 18 ? 10 : gid + 2);
                auto prev_cell = gid == 0 ? 8 : (gid == 10 ? 18 : gid - 2);
                conns.push_back(gap_junction_connection({next_cell, 0}, {gid, 0}, 0.03));
                conns.push_back(gap_junction_connection({prev_cell, 0}, {gid, 0}, 0.03));
            }
            return conns;
        }

    protected:
        cell_size_type n_ = 20;
    };
    execution_context context;
    fvm_cell fvcell(context);

    gap_recipe rec;
    std::vector<cable_cell> cell_group0;
    std::vector<cable_cell> cell_group1;

    // Make 20 cells
    for (unsigned i = 0; i < 20; i++) {
        cable_cell c;
        c.add_soma(2.1);
        if (i % 2 == 0) {
            c.add_gap_junction({0, 1});
        }
        if (i < 10) {
            cell_group0.push_back(std::move(c));
        }
        else {
            cell_group1.push_back(std::move(c));
        }
    }

    std::vector<cell_gid_type> gids_cg0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<cell_gid_type> gids_cg1 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    std::vector<fvm_index_type> cell_to_intdom0, cell_to_intdom1;

    auto num_dom0 = fvcell.fvm_intdom(rec, gids_cg0, cell_to_intdom0);
    auto num_dom1 = fvcell.fvm_intdom(rec, gids_cg1, cell_to_intdom1);

    fvm_discretization D0 = fvm_discretize(cell_group0, neuron_parameter_defaults);
    fvm_discretization D1 = fvm_discretize(cell_group1, neuron_parameter_defaults);

    auto GJ0 = fvcell.fvm_gap_junctions(cell_group0, gids_cg0, rec, D0);
    auto GJ1 = fvcell.fvm_gap_junctions(cell_group1, gids_cg1, rec, D1);

    EXPECT_EQ(10u, GJ0.size());
    EXPECT_EQ(10u, GJ1.size());

    std::vector<pair> expected_loc = {{0, 2}, {0, 8}, {2, 4}, {2, 0}, {4, 6} ,{4, 2}, {6, 8}, {6, 4}, {8, 0}, {8, 6}};

    for (unsigned i = 0; i < GJ0.size(); i++) {
        EXPECT_EQ(expected_loc[i], GJ0[i].loc);
        EXPECT_EQ(expected_loc[i], GJ1[i].loc);
    }

    std::vector<fvm_index_type> expected_doms= {0u, 1u, 0u, 2u, 0u, 3u, 0u, 4u, 0u, 5u};
    EXPECT_EQ(6u, num_dom0);
    EXPECT_EQ(6u, num_dom1);

    EXPECT_EQ(expected_doms, cell_to_intdom0);
    EXPECT_EQ(expected_doms, cell_to_intdom1);

}

TEST(fvm_lowered, integration_domains) {
    {
        execution_context context;
        fvm_cell fvcell(context);

        std::vector<cell_gid_type> gids = {11u, 5u, 2u, 3u, 0u, 8u, 7u};
        std::vector<fvm_index_type> cell_to_intdom;

        auto num_dom = fvcell.fvm_intdom(gap_recipe_0(), gids, cell_to_intdom);
        std::vector<fvm_index_type> expected_doms= {0u, 1u, 2u, 2u, 1u, 3u, 2u};

        EXPECT_EQ(4u, num_dom);
        EXPECT_EQ(expected_doms, cell_to_intdom);
    }
    {
        execution_context context;
        fvm_cell fvcell(context);

        std::vector<cell_gid_type> gids = {11u, 5u, 2u, 3u, 0u, 8u, 7u};
        std::vector<fvm_index_type> cell_to_intdom;

        auto num_dom = fvcell.fvm_intdom(gap_recipe_1(), gids, cell_to_intdom);
        std::vector<fvm_index_type> expected_doms= {0u, 1u, 2u, 3u, 4u, 5u, 6u};

        EXPECT_EQ(7u, num_dom);
        EXPECT_EQ(expected_doms, cell_to_intdom);
    }
    {
        execution_context context;
        fvm_cell fvcell(context);

        std::vector<cell_gid_type> gids = {5u, 2u, 3u, 0u};
        std::vector<fvm_index_type> cell_to_intdom;

        auto num_dom = fvcell.fvm_intdom(gap_recipe_2(), gids, cell_to_intdom);
        std::vector<fvm_index_type> expected_doms= {0u, 0u, 0u, 0u};

        EXPECT_EQ(1u, num_dom);
        EXPECT_EQ(expected_doms, cell_to_intdom);
    }
}

