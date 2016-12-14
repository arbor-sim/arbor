#include "../gtest.h"

// Prototype mechanisms in tests
#include "../mechanisms/multicore/expsyn.hpp"
#include "../mechanisms/multicore/exp2syn.hpp"
#include "../mechanisms/multicore/hh.hpp"
#include "../mechanisms/multicore/pas.hpp"

// modcc generated mechanisms
#include "mechanisms/multicore/expsyn.hpp"
#include "mechanisms/multicore/exp2syn.hpp"
#include "mechanisms/multicore/hh.hpp"
#include "mechanisms/multicore/pas.hpp"

#include <initializer_list>
#include <backends/fvm_multicore.hpp>
#include <ion.hpp>
#include <matrix.hpp>
#include <memory/wrappers.hpp>
#include <util/rangeutil.hpp>
#include <util/cycle.hpp>

TEST(mechanisms, helpers) {
    using namespace nest::mc;
    using size_type = multicore::backend::size_type;
    using value_type = multicore::backend::value_type;

    // verify that the hh and pas channels are available
    EXPECT_TRUE(multicore::backend::has_mechanism("hh"));
    EXPECT_TRUE(multicore::backend::has_mechanism("pas"));

    std::vector<size_type> parent_index = {0,0,1,2,3,4,0,6,7,8};
    auto node_indices = std::vector<size_type>{0,6,7,8,9};
    auto weights = std::vector<value_type>(node_indices.size(), 1.0);
    auto n = node_indices.size();

    multicore::backend::array vec_i(n, 0.);
    multicore::backend::array vec_v(n, 0.);

    auto mech = multicore::backend::make_mechanism(
            "hh", memory::make_view(vec_v), memory::make_view(vec_i), weights, node_indices);

    EXPECT_EQ(mech->name(), "hh");
    EXPECT_EQ(mech->size(), 5u);

    // check that an out_of_range exception is thrown if an invalid mechanism is requested
    ASSERT_THROW(
        multicore::backend::make_mechanism("dachshund", vec_v, vec_i, weights, node_indices),
        std::out_of_range
    );
}

// Setup and update mechanism
template<typename T>
void mech_update(T* mech, int num_iters) {

    using namespace nest::mc;
    std::map<mechanisms::ionKind, mechanisms::ion<typename T::backend>> ions;

    mech->set_params(2., 0.1);
    mech->nrn_init();
    for (auto ion_kind : mechanisms::ion_kinds()) {
        auto ion_indexes = util::make_copy<std::vector<typename T::size_type>>(
            mech->node_index_
        );

        // Create and fill in the ion
        mechanisms::ion<typename T::backend> ion = ion_indexes;

        memory::fill(ion.current(), 5.);
        memory::fill(ion.reversal_potential(), 100.);
        memory::fill(ion.internal_concentration(), 10.);
        memory::fill(ion.external_concentration(), 140.);
        ions[ion_kind] = ion;

        if (mech->uses_ion(ion_kind)) {
            mech->set_ion(ion_kind, ions[ion_kind], ion_indexes);
        }
    }

    for (auto i = 0; i < mech->node_index_.size(); ++i) {
        mech->net_receive(i, 1.);
    }

    for (auto i = 0; i < num_iters; ++i) {
        mech->nrn_current();
        mech->nrn_state();
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

    auto num_syn = 32;

    typename mechanism_type::iarray indexes(num_syn);
    typename mechanism_type::array  voltage(num_syn, -65.0);
    typename mechanism_type::array  current(num_syn,   1.0);
    typename mechanism_type::array  weights(num_syn,   1.0);

    array_init(voltage, nest::mc::util::cyclic_view({ -65.0, -61.0, -63.0 }));
    array_init(current, nest::mc::util::cyclic_view({   1.0,   0.9,   1.1 }));
    array_init(weights, nest::mc::util::cyclic_view({ 1.0 }));

    // Initialise indexes
    std::vector<int> index_freq;
    if (TypeParam::index_aliasing) {
        index_freq.assign({ 4, 2, 3 });
    }
    else {
        index_freq.assign({ 1 });
    }

    auto freq_begin = nest::mc::util::cyclic_view(index_freq).cbegin();
    auto freq = freq_begin;
    auto index = indexes.begin();
    while (index != indexes.end()) {
        for (auto i = 0; i < *freq && index != indexes.end(); ++i) {
            *index++ = freq - freq_begin;
        }
        ++freq;
    }


    // Copy indexes, voltage and current to use for the prototype mechanism
    typename mechanism_type::iarray indexes_copy(indexes);
    typename mechanism_type::array  voltage_copy(voltage);
    typename mechanism_type::array  current_copy(current);
    typename mechanism_type::array  weights_copy(weights);

    // Create mechanisms
    auto mech = nest::mc::mechanisms::make_mechanism<mechanism_type>(
        voltage, current, std::move(weights), std::move(indexes)
    );

    auto mech_proto = nest::mc::mechanisms::make_mechanism<proto_mechanism_type>(
        voltage_copy, current_copy,
        std::move(weights_copy), std::move(indexes_copy)
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
        nest::mc::mechanisms::hh::mechanism_hh<nest::mc::multicore::backend>,
        nest::mc::mechanisms::hh_proto::mechanism_hh<nest::mc::multicore::backend>,
        true
   >,
    mechanism_info<
        nest::mc::mechanisms::pas::mechanism_pas<nest::mc::multicore::backend>,
        nest::mc::mechanisms::pas_proto::mechanism_pas<nest::mc::multicore::backend>
    >,
    mechanism_info<
        nest::mc::mechanisms::expsyn::mechanism_expsyn<nest::mc::multicore::backend>,
        nest::mc::mechanisms::expsyn_proto::mechanism_expsyn<nest::mc::multicore::backend>,
        true
    >,
    mechanism_info<
        nest::mc::mechanisms::exp2syn::mechanism_exp2syn<nest::mc::multicore::backend>,
        nest::mc::mechanisms::exp2syn_proto::mechanism_exp2syn<nest::mc::multicore::backend>,
        true
    >
>;

INSTANTIATE_TYPED_TEST_CASE_P(mechanism_types, mechanisms, mechanism_types);
