#include "../gtest.h"

#include <matrix.hpp>
#include <backends/catalogue_multicore.hpp>

TEST(mechanisms, helpers) {
    using namespace nest::mc;
    using memory_traits = multicore::memory_traits;
    using size_type = memory_traits::size_type;
    using catalogue = multicore::catalogue;

    // verify that the hh and pas channels are available
    EXPECT_TRUE(catalogue::has("hh"));
    EXPECT_TRUE(catalogue::has("pas"));

    std::vector<size_type> parent_index = {0,0,1,2,3,4,0,6,7,8};
    auto node_indices = std::vector<size_type>{0,6,7,8,9};
    auto n = node_indices.size();

    memory_traits::array vec_i(n, 0.);
    memory_traits::array vec_v(n, 0.);

    auto mech = catalogue::make(
            "hh", memory::make_view(vec_v), memory::make_view(vec_i), node_indices);

    EXPECT_EQ(mech->name(), "hh");
    EXPECT_EQ(mech->size(), 5u);

    // check that an out_of_range exception is thrown if an invalid mechanism is requested
    ASSERT_THROW(
        catalogue::make("dachshund", vec_v, vec_i, node_indices),
        std::out_of_range
    );
                                   //0 1 2 3 4 5 6 7 8 9
}
