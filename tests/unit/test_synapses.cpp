#include "../gtest.h"

#include <cmath>
#include <tuple>
#include <vector>

#include <cell.hpp>
#include <constants.hpp>
#include <mechcat.hpp>
#include <backends/multicore/fvm.hpp>
#include <backends/multicore/mechanism.hpp>
#include <util/optional.hpp>
#include <util/maputil.hpp>
#include <util/range.hpp>
#include <util/xtuple.hpp>

#include "common.hpp"
#include "../test_util.hpp"

using namespace arb;

using backend = ::arb::multicore::backend;
using shared_state = backend::shared_state;
using value_type = backend::value_type;
using size_type = backend::size_type;

// Access to mechanisms protected data:
using field_table_type = std::vector<util::xtuple<const char*, value_type**, value_type>>;
ACCESS_BIND(field_table_type (multicore::mechanism::*)(), field_table_ptr, &multicore::mechanism::field_table)

util::range<const value_type*> mechanism_field(std::unique_ptr<multicore::mechanism>& m, const std::string& key) {
    if (auto opt_ptr = util::value_by_key((m.get()->*field_table_ptr)(), key)) {
        const value_type* field = *opt_ptr.value();
        return util::make_range(field, field+m->size());
    }
    throw std::logic_error("internal error: no such field in mechanism");
}

ACCESS_BIND(const value_type* multicore::mechanism::*, vec_v_ptr, &multicore::mechanism::vec_v_)
ACCESS_BIND(value_type* multicore::mechanism::*, vec_i_ptr, &multicore::mechanism::vec_i_)

TEST(synapses, add_to_cell) {
    using namespace arb;

    ::arb::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    cell.add_synapse({0, 0.1}, "expsyn");
    cell.add_synapse({1, 0.2}, "exp2syn");
    cell.add_synapse({0, 0.3}, "expsyn");

    EXPECT_EQ(3u, cell.synapses().size());
    const auto& syns = cell.synapses();

    EXPECT_EQ(syns[0].location.segment, 0u);
    EXPECT_EQ(syns[0].location.position, 0.1);
    EXPECT_EQ(syns[0].mechanism.name(), "expsyn");

    EXPECT_EQ(syns[1].location.segment, 1u);
    EXPECT_EQ(syns[1].location.position, 0.2);
    EXPECT_EQ(syns[1].mechanism.name(), "exp2syn");

    EXPECT_EQ(syns[2].location.segment, 0u);
    EXPECT_EQ(syns[2].location.position, 0.3);
    EXPECT_EQ(syns[2].mechanism.name(), "expsyn");
}

template <typename Seq>
static bool all_equal_to(const Seq& s, double v) {
    return util::all_of(s, [v](double x) {
        return (std::isnan(v) && std::isnan(x)) || v==x;
    });
}

TEST(synapses, syn_basic_state) {
    using util::fill;
    using value_type = multicore::backend::value_type;
    using index_type = multicore::backend::index_type;

    int num_syn = 4;
    int num_comp = 4;
    int num_cell = 1;

    auto multicore_mechanism_instance = [](const char* name) {
        return std::unique_ptr<multicore::mechanism>(
            dynamic_cast<multicore::mechanism*>(
                global_default_catalogue().instance<backend>(name).release()));
    };

    auto expsyn = multicore_mechanism_instance("expsyn");
    ASSERT_TRUE(expsyn);

    auto exp2syn = multicore_mechanism_instance("exp2syn");
    ASSERT_TRUE(exp2syn);

    auto align = std::max(expsyn->data_alignment(), exp2syn->data_alignment());
    shared_state state(num_cell, std::vector<size_type>(num_comp, 0), align);

    state.reset(-65., constant::hh_squid_temp);
    fill(state.current_density, 1.0);
    fill(state.time_to, 0.1);
    state.set_dt();

    std::vector<index_type> syn_cv(num_syn, 0);
    std::vector<value_type> syn_weight(num_syn, 1.0);

    expsyn->instantiate(0, state, {syn_cv, syn_weight});
    exp2syn->instantiate(1, state, {syn_cv, syn_weight});

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

    expsyn->nrn_init();
    EXPECT_TRUE(all_equal_to(mechanism_field(expsyn, "g"), 0.));

    exp2syn->nrn_init();
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

