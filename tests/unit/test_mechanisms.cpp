#include "../gtest.h"

#include <matrix.hpp>
#include <backends/fvm_multicore.hpp>

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
