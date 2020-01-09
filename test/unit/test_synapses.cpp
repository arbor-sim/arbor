#include "../gtest.h"

#include <cmath>
#include <tuple>
#include <vector>

#include <arbor/constants.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/optional.hpp>
#include <arbor/cable_cell.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/multicore/mechanism.hpp"
#include "util/maputil.hpp"
#include "util/range.hpp"

#include "../common_cells.hpp"
#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;

using backend = ::arb::multicore::backend;
using shared_state = backend::shared_state;
using value_type = backend::value_type;
using size_type = backend::size_type;

// Access to more mechanism protected data:

ACCESS_BIND(const value_type* multicore::mechanism::*, vec_v_ptr, &multicore::mechanism::vec_v_)
ACCESS_BIND(value_type* multicore::mechanism::*, vec_i_ptr, &multicore::mechanism::vec_i_)

TEST(synapses, add_to_cell) {
    using namespace arb;

    auto cell = make_cell_soma_only(false);

    cell.place(mlocation{0, 0.1}, "expsyn");
    cell.place(mlocation{0, 0.2}, "exp2syn");
    cell.place(mlocation{0, 0.3}, "expsyn");

    auto syns = cell.synapses();

    ASSERT_EQ(2u, syns["expsyn"].size());
    ASSERT_EQ(1u, syns["exp2syn"].size());

    EXPECT_EQ((mlocation{0, 0.1}), syns["expsyn"][0].loc);
    EXPECT_EQ("expsyn", syns["expsyn"][0].item.name());

    EXPECT_EQ((mlocation{0, 0.3}), syns["expsyn"][1].loc);
    EXPECT_EQ("expsyn", syns["expsyn"][1].item.name());

    EXPECT_EQ((mlocation{0, 0.2}), syns["exp2syn"][0].loc);
    EXPECT_EQ("exp2syn", syns["exp2syn"][0].item.name());

    // adding a synapse to an invalid branch location should throw.
    EXPECT_THROW(cell.place(mlocation{1, 0.3}, "expsyn"), std::runtime_error);
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
    using value_type = fvm_value_type;
    using index_type = fvm_index_type;

    int num_syn = 4;
    int num_comp = 4;
    int num_intdom = 1;

    value_type temp_K = *neuron_parameter_defaults.temperature_K;

    auto expsyn = unique_cast<multicore::mechanism>(global_default_catalogue().instance<backend>("expsyn").mech);
    ASSERT_TRUE(expsyn);

    auto exp2syn = unique_cast<multicore::mechanism>(global_default_catalogue().instance<backend>("exp2syn").mech);
    ASSERT_TRUE(exp2syn);

    std::vector<fvm_gap_junction> gj = {};
    auto align = std::max(expsyn->data_alignment(), exp2syn->data_alignment());

    shared_state state(num_intdom,
        std::vector<index_type>(num_comp, 0),
        {},
        std::vector<value_type>(num_comp, -65),
        std::vector<value_type>(num_comp, temp_K),
        std::vector<value_type>(num_comp, 1.),
        align);

    state.reset();
    fill(state.current_density, 1.0);
    fill(state.time_to, 0.1);
    state.set_dt();

    std::vector<index_type> syn_cv(num_syn, 0);
    std::vector<index_type> syn_mult(num_syn, 1);
    std::vector<value_type> syn_weight(num_syn, 1.0);

    expsyn->instantiate(0, state, {}, {syn_cv, syn_weight, syn_mult});
    exp2syn->instantiate(1, state, {}, {syn_cv, syn_weight, syn_mult});

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
    v_ptr = expsyn.get()->*vec_v_ptr;
    EXPECT_TRUE(all_equal_to(util::make_range(v_ptr, v_ptr+num_comp), -65.));

    v_ptr = exp2syn.get()->*vec_v_ptr;
    EXPECT_TRUE(all_equal_to(util::make_range(v_ptr, v_ptr+num_comp), -65.));

    const value_type* i_ptr;
    i_ptr = expsyn.get()->*vec_i_ptr;
    EXPECT_TRUE(all_equal_to(util::make_range(i_ptr, i_ptr+num_comp), 1.));

    i_ptr = exp2syn.get()->*vec_i_ptr;
    EXPECT_TRUE(all_equal_to(util::make_range(i_ptr, i_ptr+num_comp), 1.));

    // Initialize state then check g, A, B have been set to zero.

    expsyn->initialize();
    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "g"), 0.));

    exp2syn->initialize();
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "A"), 0.));
    EXPECT_TRUE(all_equal_to(mechanism_field(exp2syn, "B"), 0.));

    // Deliver two events (at time 0), one each to expsyn synapses 1 and 3
    // and exp2syn synapses 0 and 2.

    std::vector<deliverable_event> events = {
        {0., {0, 1, 0}, 3.14f},
        {0., {0, 3, 0}, 1.41f},
        {0., {1, 0, 0}, 2.71f},
        {0., {1, 2, 0}, 0.07f}
    };
    state.deliverable_events.init(events);
    state.deliverable_events.mark_until_after(state.time);

    expsyn->deliver_events();
    exp2syn->deliver_events();

    using fvec = std::vector<fvm_value_type>;

    EXPECT_TRUE(testing::seq_almost_eq<fvm_value_type>(
        fvec({0, 3.14f, 0, 1.41f}), mechanism_field(expsyn, "g")));

    double factor = mechanism_field(exp2syn, "factor")[0];
    EXPECT_TRUE(factor>1.);
    fvec expected = {2.71f*factor, 0, 0.07f*factor, 0};

    EXPECT_TRUE(testing::seq_almost_eq<fvm_value_type>(expected, mechanism_field(exp2syn, "A")));
    EXPECT_TRUE(testing::seq_almost_eq<fvm_value_type>(expected, mechanism_field(exp2syn, "B")));
}

