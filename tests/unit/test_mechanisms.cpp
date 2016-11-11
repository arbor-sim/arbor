#include "gtest.h"

// Prototype mechanisms in tests
#include "../mechanisms/expsyn.hpp"
#include "../mechanisms/exp2syn.hpp"
#include "../mechanisms/pas.hpp"

//#include "../src/mechanism_interface.hpp"
#include "mechanism_catalogue.hpp"
#include "matrix.hpp"

TEST(mechanisms, helpers) {
    using namespace nest::mc;
    using catalogue = mechanisms::catalogue<double, int>;

    // verify that the hh and pas channels are available
    EXPECT_TRUE(catalogue::has("hh"));
    EXPECT_TRUE(catalogue::has("pas"));

    std::vector<int> parent_index = {0,0,1,2,3,4,0,6,7,8};
    memory::HostVector<int> node_indices = std::vector<int>{0,6,7,8,9};
    auto n = node_indices.size();

    //nest::mc::matrix<double, int> matrix(parent_index);
    memory::HostVector<double> vec_i(n, 0.);
    memory::HostVector<double> vec_v(n, 0.);

    auto mech = catalogue::make("hh", vec_v, vec_i, node_indices);

    EXPECT_EQ(mech->name(), "hh");
    EXPECT_EQ(mech->size(), 5u);
    //EXPECT_EQ(mech->matrix_, &matrix);

    // check that an out_of_range exception is thrown if an invalid mechanism is
    // requested
    ASSERT_THROW(
        catalogue::make("dachshund", vec_v, vec_i, node_indices),
        std::out_of_range
    );
                                   //0 1 2 3 4 5 6 7 8 9
}

// Setup and update mechanism
template<typename T>
void mech_update(T* mech, const typename T::vector_type& areas, int num_iters) {
    mech->set_areas(areas);
    mech->set_params(2., 0.1);
    mech->nrn_init();
    for (auto i = 0; i < mech->node_index_.size(); ++i) {
        mech->net_receive(i, 1.);
    }

    for (auto i = 0; i < num_iters; ++i) {
        mech->nrn_current();
        mech->nrn_state();
    }
}

template<typename S, typename T, int freq=1>
struct mechanism_info {
    using mechanism_type = S;
    using proto_mechanism_type = T;
    static constexpr int index_freq = freq;
};

template<typename T>
class mechanisms : public ::testing::Test { };

TYPED_TEST_CASE_P(mechanisms);

TYPED_TEST_P(mechanisms, update) {
    using mechanism_type = typename TypeParam::mechanism_type;
    using proto_mechanism_type = typename TypeParam::proto_mechanism_type;

    // Type checking
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::index_type,
                              typename mechanism_type::index_type>::value));
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::value_type,
                              typename mechanism_type::value_type>::value));
    EXPECT_TRUE((std::is_same<typename proto_mechanism_type::vector_type,
                              typename mechanism_type::vector_type>::value));

    auto num_syn = 32;

    // Indexes are aliased
    typename mechanism_type::index_type indexes(num_syn);
    typename mechanism_type::vector_type voltage(num_syn, -65.0);
    typename mechanism_type::vector_type current(num_syn,   1.0);
    typename mechanism_type::vector_type areas(num_syn, 2);

    // Initialise indexes
    for (auto i = 0; i < num_syn; ++i) {
        indexes[i] = i / TypeParam::index_freq;
    }

    auto mech = nest::mc::mechanisms::make_mechanism<mechanism_type>(
        voltage, current, indexes
    );

    // Create a prototype mechanism that we will check against it
    auto indexes_copy = indexes;
    auto voltage_copy = voltage;
    auto current_copy = current;
    auto mech_proto = nest::mc::mechanisms::make_mechanism<proto_mechanism_type>(
        voltage_copy, current_copy, indexes_copy
    );

    mech_update(dynamic_cast<mechanism_type*>(mech.get()), areas, 10);
    mech_update(dynamic_cast<proto_mechanism_type*>(mech_proto.get()), areas, 10);

    auto citer = current_copy.begin();
    for (auto const& c: current) {
        EXPECT_NEAR(*citer++, c, 1e-6);
    }
}

REGISTER_TYPED_TEST_CASE_P(mechanisms, update);

using mechanism_types = ::testing::Types<
    mechanism_info<
        nest::mc::mechanisms::pas::mechanism_pas<double, int>,
        nest::mc::mechanisms::pas_test::mechanism_pas<double, int>
    >,
    mechanism_info<
        nest::mc::mechanisms::expsyn::mechanism_expsyn<double, int>,
        nest::mc::mechanisms::expsyn_test::mechanism_expsyn<double, int>,
        2
    >,
    mechanism_info<
        nest::mc::mechanisms::exp2syn::mechanism_exp2syn<double, int>,
        nest::mc::mechanisms::exp2syn_test::mechanism_exp2syn<double, int>,
        2
    >
>;

INSTANTIATE_TYPED_TEST_CASE_P(mechanism_types, mechanisms, mechanism_types);
