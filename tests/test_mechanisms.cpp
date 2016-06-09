#include "gtest.h"

#include "../src/mechanism_interface.hpp"
#include "../src/matrix.hpp"

TEST(mechanisms, helpers) {
    nest::mc::mechanisms::setup_mechanism_helpers();

    EXPECT_EQ(nest::mc::mechanisms::mechanism_helpers.size(), 2u);

    // verify that the hh and pas channels are available
    EXPECT_EQ(nest::mc::mechanisms::get_mechanism_helper("hh")->name(), "hh");
    EXPECT_EQ(nest::mc::mechanisms::get_mechanism_helper("pas")->name(), "pas");

    // check that an out_of_range exception is thrown if an invalid mechanism is
    // requested
    ASSERT_THROW(
        nest::mc::mechanisms::get_mechanism_helper("dachshund"),
        std::out_of_range
    );

                                   //0 1 2 3 4 5 6 7 8 9
    std::vector<int> parent_index = {0,0,1,2,3,4,0,6,7,8};
    memory::HostVector<int> node_indices = std::vector<int>{0,6,7,8,9};
    auto n = node_indices.size();

    //nest::mc::matrix<double, int> matrix(parent_index);
    memory::HostVector<double> vec_i(n, 0.);
    memory::HostVector<double> vec_v(n, 0.);

    auto& helper = nest::mc::mechanisms::get_mechanism_helper("hh");
    auto mech = helper->new_mechanism(vec_v, vec_i, node_indices);

    EXPECT_EQ(mech->name(), "hh");
    EXPECT_EQ(mech->size(), 5u);
    //EXPECT_EQ(mech->matrix_, &matrix);
}
