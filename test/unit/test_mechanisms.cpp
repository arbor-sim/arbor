#include <gtest/gtest.h>

// TODO: Amend for new mechanism architecture
#if 0

// Prototype mechanisms in tests
#include "mech_proto/expsyn_cpu.hpp"
#include "mech_proto/exp2syn_cpu.hpp"
#include "mech_proto/hh_cpu.hpp"
#include "mech_proto/pas_cpu.hpp"
#include "mech_proto/test_kin1_cpu.hpp"
#include "mech_proto/test_kinlva_cpu.hpp"
#include "mech_proto/test_ca_cpu.hpp"

// modcc generated mechanisms
#include "mechanisms/multicore/expsyn_cpu.hpp"
#include "mechanisms/multicore/exp2syn_cpu.hpp"
#include "mechanisms/multicore/hh_cpu.hpp"
#include "mechanisms/multicore/pas_cpu.hpp"
#include "mechanisms/multicore/test_kin1_cpu.hpp"
#include "mechanisms/multicore/test_kinlva_cpu.hpp"
#include "mechanisms/multicore/test_ca_cpu.hpp"

#include <initializer_list>
#include <backends/multicore/fvm.hpp>
#include <ion.hpp>
#include <matrix.hpp>
#include <memory/wrappers.hpp>
#include <util/rangeutil.hpp>
#include <util/cycle.hpp>

TEST(mechanisms, helpers) {
    using namespace arb;
    using size_type = multicore::backend::size_type;
    using value_type = multicore::backend::value_type;

    // verify that the hh and pas channels are available
    EXPECT_TRUE(multicore::backend::has_mechanism("hh"));
    EXPECT_TRUE(multicore::backend::has_mechanism("pas"));

    std::vector<size_type> parent_index = {0,0,1,2,3,4,0,6,7,8};
    auto node_index = std::vector<size_type>{0,6,7,8,9};
    auto weights = std::vector<value_type>(node_index.size(), 1.0);
    auto n = node_index.size();

    // one cell
    size_type ncell = 1;
    std::vector<size_type> cell_index(n, 0u);

    multicore::backend::array vec_i(n, 0.);
    multicore::backend::array vec_v(n, 0.);
    multicore::backend::array vec_t(ncell, 0.);
    multicore::backend::array vec_t_to(ncell, 0.);
    multicore::backend::array vec_dt(n, 0.);

    auto mech = multicore::backend::make_mechanism("hh", 0,
            memory::make_view(cell_index), vec_t, vec_t_to, vec_dt,
            vec_v, vec_i, weights, node_index);

    EXPECT_EQ(mech->name(), "hh");
    EXPECT_EQ(mech->size(), 5u);

    // check that an out_of_range exception is thrown if an invalid mechanism is requested
    ASSERT_THROW(
        multicore::backend::make_mechanism("dachshund", 0,
            memory::make_view(cell_index), vec_t, vec_t_to, vec_dt,
            vec_v, vec_i, weights, node_index),
        std::out_of_range
    );
}

// Setup and update mechanism
template<typename T>
void mech_update(T* mech, unsigned num_iters) {

    using namespace arb;
    std::map<ionKind, ion<typename T::backend>> ions;

    mech->set_params();
    mech->init();
    for (auto ion_kind : ion_kinds()) {
        auto ion_indexes = util::make_copy<std::vector<typename T::size_type>>(
            mech->node_index_
        );

        // Create and fill in the ion
        ion<typename T::backend> ion = ion_indexes;

        memory::fill(ion.current(), 5.);
        memory::fill(ion.reversal_potential(), 100.);
        memory::fill(ion.internal_concentration(), 10.);
        memory::fill(ion.external_concentration(), 140.);
        ions[ion_kind] = ion;

        if (mech->uses_ion(ion_kind).uses) {
            mech->set_ion(ion_kind, ions[ion_kind], ion_indexes);
        }
    }

    for (auto i=0u; i<mech->node_index_.size(); ++i) {
        mech->net_receive(i, 1.);
    }

    for (auto i=0u; i<num_iters; ++i) {
        mech->update_current();
        mech->update_state();
    }
}

template<typename T, typename Seq>
void array_init(T& array, const Seq& seq) {
    auto seq_iter = seq.cbegin();
    for (auto& e : array) {
        e = *seq_iter++;
    }
}

template<typename S, typename T, bool alias = false>
struct mechanism_info {
    using mechanism_type = S;
    using proto_mechanism_type = T;
    static constexpr bool index_aliasing = alias;
};

template<typename T>
class mechanisms : public ::testing::Test { };

TYPED_TEST_CASE_P(mechanisms);

TYPED_TEST_P(mechanisms, update) {
    using mechanism_type = typename TypeParam::mechanism_type;
    using proto_mechanism_type = typename TypeParam::proto_mechanism_type;

    // Type checking
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::iarray,
                              typename mechanism_type::iarray>::value));
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::value_type,
                              typename mechanism_type::value_type>::value));
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::array,
                              typename mechanism_type::array>::value));

    int num_cell = 1;
    int num_syn = 32;
    int num_comp = num_syn;

    typename mechanism_type::iarray node_index(num_syn);
    typename mechanism_type::array  voltage(num_comp, -65.0);
    typename mechanism_type::array  current(num_comp,   1.0);

    typename mechanism_type::array  weights(num_syn,   1.0);

    typename mechanism_type::iarray cell_index(num_comp, 0);
    typename mechanism_type::array  time(num_cell, 2.);
    typename mechanism_type::array  time_to(num_cell, 2.1);
    typename mechanism_type::array  dt(num_comp, 2.1-2.);

    array_init(voltage, arb::util::cyclic_view({ -65.0, -61.0, -63.0 }));
    array_init(current, arb::util::cyclic_view({   1.0,   0.9,   1.1 }));
    array_init(weights, arb::util::cyclic_view({ 1.0 }));

    // Initialise indexes
    std::vector<int> index_freq;
    if (TypeParam::index_aliasing) {
        index_freq.assign({ 4, 2, 3 });
    }
    else {
        index_freq.assign({ 1 });
    }

    auto freq_begin = arb::util::cyclic_view(index_freq).cbegin();
    auto freq = freq_begin;
    auto index = node_index.begin();
    while (index != node_index.end()) {
        for (auto i = 0; i < *freq && index != node_index.end(); ++i) {
            *index++ = freq - freq_begin;
        }
        ++freq;
    }

    // Copy indexes, voltage and current to use for the prototype mechanism
    typename mechanism_type::iarray node_index_copy(node_index);
    typename mechanism_type::array  voltage_copy(voltage);
    typename mechanism_type::array  current_copy(current);
    typename mechanism_type::array  weights_copy(weights);

    // Create mechanisms
    auto mech = arb::make_mechanism<mechanism_type>(
        0, cell_index, time, time_to, dt,
        voltage, current, std::move(weights), std::move(node_index)
    );

    auto mech_proto = arb::make_mechanism<proto_mechanism_type>(
        0, cell_index, time, time_to, dt,
        voltage_copy, current_copy,
        std::move(weights_copy), std::move(node_index_copy)
    );

    mech_update(dynamic_cast<mechanism_type*>(mech.get()), 10);
    mech_update(dynamic_cast<proto_mechanism_type*>(mech_proto.get()), 10);

    auto citer = current_copy.begin();
    for (auto const& c: current) {
        EXPECT_NEAR(*citer++, c, 1e-6);
    }
}

REGISTER_TYPED_TEST_CASE_P(mechanisms, update);

using mechanism_types = ::testing::Types<
    mechanism_info<
        arb::multicore::mechanism_hh<arb::multicore::backend>,
        arb::multicore::mechanism_hh_proto<arb::multicore::backend>
    >,
    mechanism_info<
        arb::multicore::mechanism_pas<arb::multicore::backend>,
        arb::multicore::mechanism_pas_proto<arb::multicore::backend>
    >,
    mechanism_info<
        arb::multicore::mechanism_expsyn<arb::multicore::backend>,
        arb::multicore::mechanism_expsyn_proto<arb::multicore::backend>,
        true
    >,
    mechanism_info<
        arb::multicore::mechanism_exp2syn<arb::multicore::backend>,
        arb::multicore::mechanism_exp2syn_proto<arb::multicore::backend>,
        true
    >,
    mechanism_info<
        arb::multicore::mechanism_test_kin1<arb::multicore::backend>,
        arb::multicore::mechanism_test_kin1_proto<arb::multicore::backend>
    >,
    mechanism_info<
        arb::multicore::mechanism_test_kinlva<arb::multicore::backend>,
        arb::multicore::mechanism_test_kinlva_proto<arb::multicore::backend>
    >,
    mechanism_info<
        arb::multicore::mechanism_test_ca<arb::multicore::backend>,
        arb::multicore::mechanism_test_ca_proto<arb::multicore::backend>
    >
>;

INSTANTIATE_TYPED_TEST_CASE_P(mechanism_types, mechanisms, mechanism_types);

#endif // 0
