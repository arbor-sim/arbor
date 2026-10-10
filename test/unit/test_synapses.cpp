#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/cable_cell.hpp>

#include "backends/multicore/fvm.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/shared_state.hpp"
#endif
#include "util/range.hpp"

#include "../common_cells.hpp"
#include "common.hpp"
#include "mech_private_field_access.hpp"
#include "unit_test_catalogue.hpp"

using namespace arb;

using backend = multicore::backend;
using shared_state = backend::shared_state;
using value_type = backend::value_type;
using size_type = backend::size_type;

ACCESS_BIND(arb_mechanism_ppack mechanism::*, get_ppack, &mechanism::ppack_);

#ifdef ARB_GPU_ENABLED
TEST(synapses, gpu_empty_mechanism) {
    auto thread_pool = std::make_shared<arb::threading::task_system>();
    auto syn = global_default_catalogue().instance(arb_backend_kind_gpu, "expsyn").mech;
    gpu::shared_state state(thread_pool, 1, 1, {0}, {-65}, {23}, {1}, {10}, {},
                            fvm_detector_info{}, syn->data_alignment());
    std::vector<arb_index_type> cv;
    state.instantiate(*syn, {}, mechanism_layout{cv}, {});
    syn->initialize();
    syn->update_current();
    syn->update_state();
    syn->update_ions();
}

TEST(synapses, gpu_ionic_current_accumulation) {
    auto thread_pool = std::make_shared<arb::threading::task_system>();
    auto syn = make_unit_test_catalogue().instance(arb_backend_kind_gpu, "point_ica_current").mech;
    gpu::shared_state state(thread_pool, 1, 4, {0, 0, 0, 0}, {-65, -65, -65, -65},
                            {23, 23, 23, 23}, {1, 1, 1, 1}, {10, 10, 10, 10}, {},
                            fvm_detector_info{}, syn->data_alignment());
    fvm_ion_config ion;
    ion.cv = {1, 3};
    ion.charge = 2;
    state.add_ion("ca", ion);
    std::vector<arb_index_type> cv(129, 1);
    std::fill(cv.begin()+64, cv.end(), 3);
    mechanism_layout layout{cv};
    layout.weight.assign(cv.size(), 0.5);
    state.instantiate(*syn, {}, layout, {});
    state.reset();
    syn->initialize();
    for (unsigned repeat = 0; repeat<20; ++repeat) {
        std::vector<double> currents(cv.size());
        double expected[2] = {};
        for (unsigned i = 0; i<cv.size(); ++i) {
            currents[i] = (repeat+1)*(.001+(1+std::sin(double(i)))*.01);
            expected[i<64? 0: 1] += 0.5*currents[i];
        }
        write_mechanism_field(syn, "ica_nA", currents);
        state.zero_currents();
        syn->update_current();
        syn->update_current();
        auto ionic = memory::on_host(state.ion_data.at("ca").iX_);
        auto total = memory::on_host(state.current_density);
        EXPECT_EQ(0, total[0]);
        EXPECT_EQ(0, total[2]);
        for (unsigned i = 0; i<2; ++i) {
            EXPECT_NEAR(2*expected[i], ionic[i], 1e-12);
            EXPECT_DOUBLE_EQ(ionic[i], total[ion.cv[i]]);
        }
    }
}

TEST(synapses, gpu_current_reproducibility) {
    auto thread_pool = std::make_shared<arb::threading::task_system>();
    auto syn = global_default_catalogue().instance(arb_backend_kind_gpu, "expsyn").mech;
    gpu::shared_state state(thread_pool, 1, 3, {0, 0, 0}, {-65, -65, -65},
                            {23, 23, 23}, {1, 1, 1}, {10, 10, 10}, {},
                            fvm_detector_info{}, syn->data_alignment());
    state.reset();
    constexpr unsigned n = 8193;
    std::vector<arb_index_type> cv(n, 0);
    std::fill(cv.end()-129, cv.end(), 2);
    mechanism_layout layout{cv};
    layout.weight.assign(n, 1.0);
    state.instantiate(*syn, {}, layout, {});
    syn->initialize();

    // Change state between batches to catch stale/unwritten scratch entries too.
    for (double scale: {1.0, 2.0}) {
        std::vector<double> g(n), expected_i(3), expected_g(3);
        for (unsigned i = 0; i<n; ++i) {
            g[i] = scale*(0.001+(1+std::sin(double(i)))*0.01);
            expected_i[layout.cv[i]] += -65*g[i];
            expected_g[layout.cv[i]] += g[i];
        }
        write_mechanism_field(syn, "g", g);
        std::vector<double> first_i, first_g;
        for (unsigned repeat = 0; repeat<30; ++repeat) {
            state.zero_currents();
            syn->update_current();
            auto currents = memory::on_host(state.current_density);
            auto conductances = memory::on_host(state.conductivity);
            std::vector<double> actual_i(currents.begin(), currents.end());
            std::vector<double> actual_g(conductances.begin(), conductances.end());
            for (unsigned cv = 0; cv<3; ++cv) {
                EXPECT_NEAR(expected_i[cv], actual_i[cv], 1e-10);
                EXPECT_NEAR(expected_g[cv], actual_g[cv], 1e-10);
            }
            if (!repeat) { first_i = actual_i; first_g = actual_g; }
            EXPECT_EQ(first_i, actual_i);
            EXPECT_EQ(first_g, actual_g);
        }
    }
}
#endif

TEST(synapses, add_to_cell) {
    using namespace arb;

    auto description = make_cell_soma_only(false);

    description.decorations.place(mlocation{0, 0.1}, synapse("expsyn"), "synapse0");
    description.decorations.place(mlocation{0, 0.2}, synapse("exp2syn"), "synapse1");
    description.decorations.place(mlocation{0, 0.3}, synapse("expsyn"), "synapse2");

    cable_cell cell(description);

    auto syns = cell.synapses();

    ASSERT_EQ(2u, syns["expsyn"].size());
    ASSERT_EQ(1u, syns["exp2syn"].size());

    EXPECT_EQ((mlocation{0, 0.1}), syns["expsyn"][0].loc);
    EXPECT_EQ("expsyn", syns["expsyn"][0].item.mech.name());

    EXPECT_EQ((mlocation{0, 0.3}), syns["expsyn"][1].loc);
    EXPECT_EQ("expsyn", syns["expsyn"][1].item.mech.name());

    EXPECT_EQ((mlocation{0, 0.2}), syns["exp2syn"][0].loc);
    EXPECT_EQ("exp2syn", syns["exp2syn"][0].item.mech.name());

    // adding a synapse to an invalid branch location should throw.
    description.decorations.place(mlocation{1, 0.3}, synapse("expsyn"), "synapse3");
    EXPECT_THROW((cell=description), std::runtime_error);
}

template <typename Seq>
static bool all_equal_to(const Seq& s, double v) {
    return util::all_of(s, [v](double x) {
        return (std::isnan(v) && std::isnan(x)) || v==x;
    });
}

template <typename A, typename B>
auto unique_cast(std::unique_ptr<B> p) {
    return std::unique_ptr<A>(dynamic_cast<A*>(p.release()));
}

TEST(synapses, syn_basic_state) {
    using util::fill;
    using value_type = arb_value_type;
    using index_type = arb_index_type;

    auto thread_pool = std::make_shared<arb::threading::task_system>();

    int num_syn = 4;
    int num_comp = 4;
    int num_cells = 1;

    value_type temp_K = *neuron_parameter_defaults.temperature_K;

    auto expsyn = unique_cast<mechanism>(global_default_catalogue().instance(backend::kind, "expsyn").mech);
    ASSERT_TRUE(expsyn);

    auto exp2syn = unique_cast<mechanism>(global_default_catalogue().instance(backend::kind, "exp2syn").mech);
    ASSERT_TRUE(exp2syn);

    auto align = std::max(expsyn->data_alignment(), exp2syn->data_alignment());

    shared_state state(thread_pool,
                       num_cells,
                       num_comp,
                       std::vector<index_type>(num_comp, 0),      // cv -> cell
                       std::vector<value_type>(num_comp, -65),    // U_m
                       std::vector<value_type>(num_comp, temp_K), // T
                       std::vector<value_type>(num_comp, 1.),     // diameter
                       std::vector<value_type>(num_comp, 10.),    // area
                       std::vector<index_type>(0),                // src -> spike
                       fvm_detector_info{},
                       align);

    state.reset();
    fill(state.current_density, 1.0);
    const auto dts = timestep_range(0.1, 0.1);
    state.update_time_to(dts[0]);

    std::vector<index_type> syn_cv(num_syn, 0);
    std::vector<index_type> syn_mult(num_syn, 1);
    std::vector<value_type> syn_weight(num_syn, 1.0);

    state.instantiate(*expsyn,  {}, {syn_cv, {}, syn_weight, syn_mult}, {});
    state.instantiate(*exp2syn, {}, {syn_cv, {}, syn_weight, syn_mult}, {});

    // Parameters initialized to default values?

    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "e"),   0.));
    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "tau"), 2.0));
    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "g"),   NAN));

    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "e"),    0.));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "tau1"), 0.5));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "tau2"), 2.0));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "A"),    NAN));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "B"),    NAN));

    // Current and voltage views correctly hooked up?
    const value_type* v_ptr;
    v_ptr = (expsyn.get()->*get_ppack).vec_v;
    EXPECT_TRUE(all_equal_to(util::make_range(v_ptr, v_ptr+num_comp), -65.));

    v_ptr = (exp2syn.get()->*get_ppack).vec_v;
    EXPECT_TRUE(all_equal_to(util::make_range(v_ptr, v_ptr+num_comp), -65.));

    const value_type* i_ptr;
    i_ptr = (expsyn.get()->*get_ppack).vec_i;
    EXPECT_TRUE(all_equal_to(util::make_range(i_ptr, i_ptr+num_comp), 1.));

    i_ptr = (exp2syn.get()->*get_ppack).vec_i;
    EXPECT_TRUE(all_equal_to(util::make_range(i_ptr, i_ptr+num_comp), 1.));

    // Initialize state then check g, A, B have been set to zero.

    expsyn->initialize();
    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "g"), 0.));

    exp2syn->initialize();
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "A"), 0.));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "B"), 0.));

    // Deliver two events (at time 0), one each to expsyn synapses 1 and 3
    // and exp2syn synapses 0 and 2.
    std::vector<pse_vector> events{{{0, 0.0, 3.14f}, {1, 0.0, 1.41f}, {2, 0.0, 2.71f}, {3, 0.0, 0.07f}}};
    auto lanes = event_lane_subrange(events.begin(), events.end());
    std::vector<target_handle> handles{{0, 1}, {0, 3}, {1, 0}, {1, 2}};
    std::vector<size_t> divs{0, handles.size()};
    state.begin_epoch(lanes, {}, dts, handles, divs);
    state.mark_events();

    state.deliver_events(*expsyn);
    state.deliver_events(*exp2syn);

    using fvec = std::vector<arb_value_type>;

    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(fvec({0, 3.14f, 0, 1.41f}),
                                                       mechanism_field(expsyn, "g")));

    double factor = mechanism_field(exp2syn, "factor")[0];
    EXPECT_TRUE(factor>1.);
    fvec expected = {2.71f*factor, 0, 0.07f*factor, 0};

    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(expected, mechanism_field(exp2syn, "A")));
    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(expected, mechanism_field(exp2syn, "B")));
}

