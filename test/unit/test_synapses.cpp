#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/cable_cell.hpp>

#include "backends/multicore/fvm.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"

#include "../common_cells.hpp"
#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;

using backend = multicore::backend;
using shared_state = backend::shared_state;
using value_type = backend::value_type;
using size_type = backend::size_type;

ACCESS_BIND(arb_mechanism_ppack mechanism::*, get_ppack, &mechanism::ppack_);

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

    state.instantiate(*expsyn,  0, {}, {syn_cv, {}, syn_weight, syn_mult}, {});
    state.instantiate(*exp2syn, 1, {}, {syn_cv, {}, syn_weight, syn_mult}, {});

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

    state.begin_epoch({{{{0., {0, 1}, 3.14f}, {0., {0, 3}, 1.41f}}},  // events for mech_id == 0
                       {{{0., {1, 0}, 2.71f}, {0., {1, 2}, 0.07f}}}}, // events for mech_id == 1
                      {}, dts);
    state.mark_events();

    state.deliver_events(*expsyn);
    state.deliver_events(*exp2syn);

    using fvec = std::vector<arb_value_type>;

    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(
        fvec({0, 3.14f, 0, 1.41f}), mechanism_field(expsyn, "g")));

    double factor = mechanism_field(exp2syn, "factor")[0];
    EXPECT_TRUE(factor>1.);
    fvec expected = {2.71f*factor, 0, 0.07f*factor, 0};

    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(expected, mechanism_field(exp2syn, "A")));
    EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(expected, mechanism_field(exp2syn, "B")));
}

