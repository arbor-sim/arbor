#include "../gtest.h"

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/version.hpp>
#include <arborenv/gpu_env.hpp>

#include "backends/event.hpp"
#include "backends/multicore/fvm.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif
#include "fvm_lowered_cell_impl.hpp"
#include "memory/gpu_wrappers.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "unit_test_catalogue.hpp"
#include "../common_cells.hpp"
#include "../simple_recipes.hpp"

using namespace arb;

using multicore_fvm_cell = fvm_lowered_cell_impl<multicore::backend>;
using multicore_shared_state = multicore::backend::shared_state;
ACCESS_BIND(std::unique_ptr<multicore_shared_state> multicore_fvm_cell::*, multicore_fvm_state_ptr, &multicore_fvm_cell::state_);


template <typename Backend>
struct backend_access {
    using fvm_cell = multicore_fvm_cell;

    static multicore_shared_state& state(fvm_cell& cell) {
        return *(cell.*multicore_fvm_state_ptr).get();
    }

    static fvm_value_type deref(const fvm_value_type* p) { return *p; }
};

#ifdef ARB_GPU_ENABLED

using gpu_fvm_cell = fvm_lowered_cell_impl<gpu::backend>;
using gpu_shared_state = gpu::backend::shared_state;
ACCESS_BIND(std::unique_ptr<gpu_shared_state> gpu_fvm_cell::*, gpu_fvm_state_ptr, &gpu_fvm_cell::state_);

template <>
struct backend_access<gpu::backend> {
    using fvm_cell = gpu_fvm_cell;

    static gpu_shared_state& state(fvm_cell& cell) {
        return *(cell.*gpu_fvm_state_ptr).get();
    }

    static fvm_value_type deref(const fvm_value_type* p) {
        fvm_value_type r;
        memory::gpu_memcpy_d2h(&r, p, sizeof(r));
        return r;
    }
};

#endif

template <typename Backend>
void run_v_i_probe_test(const context& ctx) {
    using fvm_cell = typename backend_access<Backend>::fvm_cell;
    auto deref = [](const fvm_value_type* p) { return backend_access<Backend>::deref(p); };

    cable_cell bs = make_cell_ball_and_stick(false);

    i_clamp stim(0, 100, 0.3);
    bs.place(mlocation{1, 1}, stim);

    cable1d_recipe rec(bs);

    mlocation loc0{0, 0};
    mlocation loc1{1, 1};
    mlocation loc2{1, 0.3};

    rec.add_probe(0, 10, cell_probe_membrane_voltage{loc0});
    rec.add_probe(0, 20, cell_probe_membrane_voltage{loc1});
    rec.add_probe(0, 30, cell_probe_total_ionic_current_density{loc2});

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(*ctx);
    lcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    EXPECT_EQ(3u, rec.num_probes(0));
    EXPECT_EQ(3u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({0, 1}).tag);
    EXPECT_EQ(30, probe_map.at({0, 2}).tag);

    probe_handle p0 = probe_map.at({0, 0}).handle;
    probe_handle p1 = probe_map.at({0, 1}).handle;
    probe_handle p2 = probe_map.at({0, 2}).handle;

    // Expect initial probe values to be the resting potential
    // for the voltage probes (cell membrane potential should
    // be constant), and zero for the current probe.

    auto& state = backend_access<Backend>::state(lcell);
    auto& voltage = state.voltage;

    fvm_value_type resting = voltage[0];
    EXPECT_NE(0.0, resting);

    // (Probe handles are just pointers in this implementation).
    EXPECT_EQ(resting, deref(p0));
    EXPECT_EQ(resting, deref(p1));
    EXPECT_EQ(0.0, deref(p2));

    // After an integration step, expect voltage probe values
    // to differ from resting, and between each other, and
    // for there to be a non-zero current.
    //
    // First probe, at (0,0), should match voltage in first
    // compartment.

    lcell.integrate(0.01, 0.0025, {}, {});

    EXPECT_NE(resting, deref(p0));
    EXPECT_NE(resting, deref(p1));
    EXPECT_NE(deref(p0), deref(p1));
    EXPECT_NE(0.0, deref(p2));

    fvm_value_type v = voltage[0];
    EXPECT_EQ(v, deref(p0));
}

template <typename Backend>
void run_expsyn_g_probe_test(const context& ctx, bool coalesce_synapses = false) {
    using fvm_cell = typename backend_access<Backend>::fvm_cell;
    auto deref = [](const fvm_value_type* p) { return backend_access<Backend>::deref(p); };

    const double tau = 2.0;
    EXPECT_EQ(tau, global_default_catalogue()["expsyn"].parameters.at("tau").default_value);

    // Ball-and-stick cell, two synapses, both in same CV.
    mlocation loc0{1, 0.8};
    mlocation loc1{1, 1.0};

    cable_cell bs = make_cell_ball_and_stick(false);
    bs.place(loc0, "expsyn");
    bs.place(loc1, "expsyn");
    bs.default_parameters.discretization = cv_policy_fixed_per_branch(2);

    cable1d_recipe rec(bs, coalesce_synapses);
    rec.add_probe(0, 10, cell_probe_point_state{0u, "expsyn", "g"});
    rec.add_probe(0, 20, cell_probe_point_state{1u, "expsyn", "g"});

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(*ctx);
    lcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    EXPECT_EQ(2u, rec.num_probes(0));
    EXPECT_EQ(2u, probe_map.size());

    EXPECT_EQ(10, probe_map.at({0, 0}).tag);
    EXPECT_EQ(20, probe_map.at({0, 1}).tag);

    probe_handle p0 = probe_map.at({0, 0}).handle;
    probe_handle p1 = probe_map.at({0, 1}).handle;

    // Expect initial probe values to be intial synapse g == 0.

    EXPECT_EQ(0.0, deref(p0));
    EXPECT_EQ(0.0, deref(p1));

    if (coalesce_synapses) {
        // Should be the same raw pointer!
        EXPECT_EQ(p0, p1);
    }

    // Integrate to 3 ms, with one event at 1ms to first expsyn weight 0.5,
    // and another at 2ms to second, weight 1.

    std::vector<deliverable_event> evs = {
        {1.0, targets[0], 0.5},
        {2.0, targets[1], 1.0}
    };
    const double tfinal = 3.;
    const double dt = 0.001;
    lcell.integrate(tfinal, dt, evs, {});

    fvm_value_type g0 = deref(p0);
    fvm_value_type g1 = deref(p1);

    // Expected value: weight*exp(-(t_final-t_event)/tau).
    double expected_g0 = 0.5*std::exp(-(tfinal-1.0)/tau);
    double expected_g1 = 1.0*std::exp(-(tfinal-2.0)/tau);

    const double rtol = 1e-6;
    if (coalesce_synapses) {
        EXPECT_TRUE(testing::near_relative(expected_g0+expected_g1, g0, rtol));
        EXPECT_TRUE(testing::near_relative(expected_g0+expected_g1, g1, rtol));
    }
    else {
        EXPECT_TRUE(testing::near_relative(expected_g0, g0, rtol));
        EXPECT_TRUE(testing::near_relative(expected_g1, g1, rtol));
    }
}

template <typename Backend>
void run_ion_density_probe_test(const context& ctx) {
    using fvm_cell = typename backend_access<Backend>::fvm_cell;
    auto deref = [](const fvm_value_type* p) { return backend_access<Backend>::deref(p); };

    // Use test mechanism write_Xi_Xo to check ion concentration probes and
    // density mechanism state probes.

    auto cat = make_unit_test_catalogue();
    cat.derive("write_ca1", "write_Xi_Xo", {{"xi0", 1.25}, {"xo0", 1.5}, {"s0", 1.75}}, {{"x", "ca"}});
    cat.derive("write_ca2", "write_Xi_Xo", {{"xi0", 2.25}, {"xo0", 2.5}, {"s0", 2.75}}, {{"x", "ca"}});
    cat.derive("write_na3", "write_Xi_Xo", {{"xi0", 3.25}, {"xo0", 3.5}, {"s0", 3.75}}, {{"x", "na"}});

    // Simple constant diameter cable, 3 CVs.

    cable_cell cable(sample_tree({msample{{0., 0., 0., 1.}, 0}, msample{{100., 0., 0., 1.}, 0}}, {mnpos, 0u}));
    cable.default_parameters.discretization = cv_policy_fixed_per_branch(3);

    // Calcium ions everywhere, half written by write_ca1, half by write_ca2.
    // Sodium ions only on distal half.

    cable.paint(mcable{0, 0., 0.5}, "write_ca1");
    cable.paint(mcable{0, 0.5, 1.}, "write_ca2");
    cable.paint(mcable{0, 0.5, 1.}, "write_na3");

    // Place probes in each CV.

    mlocation loc0{0, 0.1};
    mlocation loc1{0, 0.5};
    mlocation loc2{0, 0.9};

    cable1d_recipe rec(cable);
    rec.catalogue() = cat;

    // Probe (0, 0): ca internal on CV 0.
    rec.add_probe(0, 0, cell_probe_ion_int_concentration{loc0, "ca"});
    // Probe (0, 1): ca internal on CV 1.
    rec.add_probe(0, 0, cell_probe_ion_int_concentration{loc1, "ca"});
    // Probe (0, 2): ca internal on CV 2.
    rec.add_probe(0, 0, cell_probe_ion_int_concentration{loc2, "ca"});

    // Probe (0, 3): ca external on CV 0.
    rec.add_probe(0, 0, cell_probe_ion_ext_concentration{loc0, "ca"});
    // Probe (0, 4): ca external on CV 1.
    rec.add_probe(0, 0, cell_probe_ion_ext_concentration{loc1, "ca"});
    // Probe (0, 5): ca external on CV 2.
    rec.add_probe(0, 0, cell_probe_ion_ext_concentration{loc2, "ca"});
 
    // Probe (0, 6): na internal on CV 0.
    rec.add_probe(0, 0, cell_probe_ion_int_concentration{loc0, "na"});
    // Probe (0, 7): na internal on CV 2.
    rec.add_probe(0, 0, cell_probe_ion_int_concentration{loc2, "na"});

    // Probe (0, 8): write_ca2 state 's' in CV 0.
    rec.add_probe(0, 0, cell_probe_density_state{loc0, "write_ca2", "s"});
    // Probe (0, 9): write_ca2 state 's' in CV 1.
    rec.add_probe(0, 0, cell_probe_density_state{loc1, "write_ca2", "s"});
    // Probe (0, 10): write_ca2 state 's' in CV 2.
    rec.add_probe(0, 0, cell_probe_density_state{loc2, "write_ca2", "s"});

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(*ctx);
    lcell.initialize({0}, rec, cell_to_intdom, targets, probe_map);

    // Should be no sodium ion instantiated on CV 0, so probe (0, 6) should
    // have been silently discared. Similarly, write_ca2 is not instantiated on
    // CV 0, and so probe (0, 8) should have been discarded. All other probes
    // should be in the map.

    EXPECT_EQ(11u, rec.num_probes(0));
    EXPECT_EQ(9u, probe_map.size());

    probe_handle ca_int_cv0 = probe_map.at({0, 0}).handle;
    probe_handle ca_int_cv1 = probe_map.at({0, 1}).handle;
    probe_handle ca_int_cv2 = probe_map.at({0, 2}).handle;
    probe_handle ca_ext_cv0 = probe_map.at({0, 3}).handle;
    probe_handle ca_ext_cv1 = probe_map.at({0, 4}).handle;
    probe_handle ca_ext_cv2 = probe_map.at({0, 5}).handle;
    EXPECT_EQ(0u, probe_map.count({0, 6}));
    probe_handle na_int_cv2 = probe_map.at({0, 7}).handle;
    EXPECT_EQ(0u, probe_map.count({0, 8}));
    probe_handle write_ca2_s_cv1 = probe_map.at({0, 9}).handle;
    probe_handle write_ca2_s_cv2 = probe_map.at({0, 10}).handle;

    // Ion concentrations should have been written in initialization.
    // For CV 1, calcium concentration should be mean of the two values
    // from write_ca1 and write_ca2.

    EXPECT_EQ(1.25, deref(ca_int_cv0));
    EXPECT_DOUBLE_EQ((1.25+2.25)/2., deref(ca_int_cv1));
    EXPECT_EQ(2.25, deref(ca_int_cv2));

    EXPECT_EQ(1.5, deref(ca_ext_cv0));
    EXPECT_DOUBLE_EQ((1.5+2.5)/2., deref(ca_ext_cv1));
    EXPECT_EQ(2.5, deref(ca_ext_cv2));

    EXPECT_EQ(3.25, deref(na_int_cv2));

    // State variable in write_ca2 should be the same in both CV 1 and 2.
    // The raw handles should be different addresses, however.

    EXPECT_EQ(2.75, deref(write_ca2_s_cv1));
    EXPECT_EQ(2.75, deref(write_ca2_s_cv2));
    EXPECT_NE(write_ca2_s_cv1, write_ca2_s_cv2);
}

template <typename Backend>
void run_ion_current_probe_test(const context& ctx) {
    using fvm_cell = typename backend_access<Backend>::fvm_cell;
    auto deref = [](const fvm_value_type* p) { return backend_access<Backend>::deref(p); };

    // Use test mechanism fixed_ica_current, and a derived mechanism for sodium, to
    // write to specific ion currents.

    auto cat = make_unit_test_catalogue();
    cat.derive("fixed_ina_current", "fixed_ica_current", {}, {{"ca", "na"}});

    cable_cell cells[2];

    // Simple constant diameter cable, 3 CVs.

    cells[0] = cable_cell(sample_tree({msample{{0., 0., 0., 1.}, 0}, msample{{100., 0., 0., 1.}, 0}}, {mnpos, 0u}));
    cells[0].default_parameters.discretization = cv_policy_fixed_per_branch(3);

    // Calcium ions everywhere, half with current density jca0, half with jca1.
    // Sodium ions only on distal half, with current densitry jna1.

    const double jca0 = 1.5; // [A/m²]
    const double jca1 = 2.0;
    const double jna1 = 2.5;

    // Scaling factor 0.1 is to convert our current densities in [A/m²] to NMODL units [mA/cm²].

    cells[0].paint(mcable{0, 0., 0.5}, mechanism_desc("fixed_ica_current").set("current_density", 0.1*jca0));
    cells[0].paint(mcable{0, 0.5, 1.}, mechanism_desc("fixed_ica_current").set("current_density", 0.1*jca1));
    cells[0].paint(mcable{0, 0.5, 1.}, mechanism_desc("fixed_ina_current").set("current_density", 0.1*jna1));

    // Make a second cable cell, with same layout but 3 times the current.

    cells[1] = cable_cell(sample_tree({msample{{0., 0., 0., 1.}, 0}, msample{{100., 0., 0., 1.}, 0}}, {mnpos, 0u}));
    cells[1].default_parameters.discretization = cv_policy_fixed_per_branch(3);

    cells[1].paint(mcable{0, 0., 0.5}, mechanism_desc("fixed_ica_current").set("current_density", 0.3*jca0));
    cells[1].paint(mcable{0, 0.5, 1.}, mechanism_desc("fixed_ica_current").set("current_density", 0.3*jca1));
    cells[1].paint(mcable{0, 0.5, 1.}, mechanism_desc("fixed_ina_current").set("current_density", 0.3*jna1));

    // Place probes in each CV on cell 0, plus one in the last CV on cell 1.

    mlocation loc0{0, 0.1};
    mlocation loc1{0, 0.5};
    mlocation loc2{0, 0.9};

    cable1d_recipe rec(cells);
    rec.catalogue() = cat;

    // Probe (0, 0): ica on CV 0.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc0, "ca"});
    // Probe (0, 1): ica on CV 1.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc1, "ca"});
    // Probe (0, 2): ica on CV 2.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc2, "ca"});

    // Probe (0, 3): ina on CV 0.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc0, "na"});
    // Probe (0, 4): ina on CV 1.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc1, "na"});
    // Probe (0, 5): ina on CV 2.
    rec.add_probe(0, 0, cell_probe_ion_current_density{loc2, "na"});

    // Probe (0, 6): total ion current density on CV 0.
    rec.add_probe(0, 0, cell_probe_total_ionic_current_density{loc0});
    // Probe (0, 7): total ion current density on CV 1.
    rec.add_probe(0, 0, cell_probe_total_ionic_current_density{loc1});
    // Probe (0, 8): total ion current density on CV 2.
    rec.add_probe(0, 0, cell_probe_total_ionic_current_density{loc2});

    // Probe (1, 0): ica on CV 5 (CV 2 of cell 1).
    rec.add_probe(1, 0, cell_probe_ion_current_density{loc2, "ca"});
    // Probe (1, 1): total ion current density on CV 5 (CV 2 of cell 1).
    rec.add_probe(1, 0, cell_probe_total_ionic_current_density{loc2});

    std::vector<target_handle> targets;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_map;

    fvm_cell lcell(*ctx);
    lcell.initialize({0, 1}, rec, cell_to_intdom, targets, probe_map);

    // Should be no sodium ion instantiated on CV 0, so probe (0, 3) should
    // have been silently discared.

    EXPECT_EQ(9u, rec.num_probes(0));
    EXPECT_EQ(2u, rec.num_probes(1));
    EXPECT_EQ(10u, probe_map.size());

    probe_handle ica_cv0 = probe_map.at({0, 0}).handle;
    probe_handle ica_cv1 = probe_map.at({0, 1}).handle;
    probe_handle ica_cv2 = probe_map.at({0, 2}).handle;
    EXPECT_EQ(0u, probe_map.count({0, 3}));
    probe_handle ina_cv1 = probe_map.at({0, 4}).handle;
    probe_handle ina_cv2 = probe_map.at({0, 5}).handle;
    probe_handle i_cv0 = probe_map.at({0, 6}).handle;
    probe_handle i_cv1 = probe_map.at({0, 7}).handle;
    probe_handle i_cv2 = probe_map.at({0, 8}).handle;

    probe_handle ica_cv5 = probe_map.at({1, 0}).handle;
    probe_handle i_cv5 = probe_map.at({1, 1}).handle;

    // Integrate cell for a bit, and check that currents add up as we expect.

    lcell.integrate(0.01, 0.0025, {}, {});

    EXPECT_DOUBLE_EQ(jca0, deref(ica_cv0));
    EXPECT_DOUBLE_EQ((jca0+jca1)/2, deref(ica_cv1));
    EXPECT_DOUBLE_EQ(jca1, deref(ica_cv2));

    EXPECT_DOUBLE_EQ(jna1/2, deref(ina_cv1));
    EXPECT_DOUBLE_EQ(jna1, deref(ina_cv2));

    EXPECT_DOUBLE_EQ(jca0, deref(i_cv0));
    EXPECT_DOUBLE_EQ(jna1/2+jca0/2+jca1/2, deref(i_cv1));
    EXPECT_DOUBLE_EQ(jna1+jca1, deref(i_cv2));

    // Currents on cell 1 should be 3 times those on cell 0.

    EXPECT_DOUBLE_EQ(jca1*3, deref(ica_cv5));
    EXPECT_DOUBLE_EQ((jna1+jca1)*3, deref(i_cv5));
}

TEST(probe, multicore_v_i) {
    context ctx = make_context();
    run_v_i_probe_test<multicore::backend>(ctx);
}

TEST(probe, multicore_expsyn_g) {
    context ctx = make_context();
    SCOPED_TRACE("uncoalesced synapses");
    run_expsyn_g_probe_test<multicore::backend>(ctx, false);
    SCOPED_TRACE("coalesced synapses");
    run_expsyn_g_probe_test<multicore::backend>(ctx, true);
}

TEST(probe, multicore_ion_conc) {
    context ctx = make_context();
    run_ion_density_probe_test<multicore::backend>(ctx);
}

TEST(probe, multicore_ion_currents) {
    context ctx = make_context();
    run_ion_current_probe_test<multicore::backend>(ctx);
}

#ifdef ARB_GPU_ENABLED
TEST(probe, gpu_v_i) {
    context ctx = make_context(proc_allocation{1, arbenv::default_gpu()});
    if (has_gpu(ctx)) {
        run_v_i_probe_test<gpu::backend>(ctx);
    }
}

TEST(probe, gpu_expsyn_g) {
    context ctx = make_context(proc_allocation{1, arbenv::default_gpu()});
    if (has_gpu(ctx)) {
        SCOPED_TRACE("uncoalesced synapses");
        run_expsyn_g_probe_test<gpu::backend>(ctx, false);
        SCOPED_TRACE("coalesced synapses");
        run_expsyn_g_probe_test<gpu::backend>(ctx, true);
    }
}

TEST(probe, gpu_ion_conc) {
    context ctx = make_context(proc_allocation{1, arbenv::default_gpu()});
    if (has_gpu(ctx)) {
        run_ion_density_probe_test<gpu::backend>(ctx);
    }
}

TEST(probe, gpu_ion_currents) {
    context ctx = make_context(proc_allocation{1, arbenv::default_gpu()});
    if (has_gpu(ctx)) {
        run_ion_current_probe_test<gpu::backend>(ctx);
    }
}
#endif

