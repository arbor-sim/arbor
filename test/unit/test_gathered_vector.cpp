#include <gtest/gtest.h>

#include <vector>

#include "arbor/common_types.hpp"
#include "communication/gathered_vector.hpp"


TEST(gathered_vector, invariants) {
    auto sc = std::vector<arb::cell_member_type> {
    {0, 0}, // 0-3
    {0, 1},
    {0, 2},
    {1, 3}, // 3-5
    {1, 4},
            // 5-5 empty!
    {3, 5}, // 5-7
    {3, 6}
    };
    auto pt = std::vector<unsigned>{0, 3, 5, 5, 7};
    auto gv = arb::gathered_vector<arb::cell_member_type>{std::move(sc), std::move(pt)};
    auto partition = gv.partition();
    auto values = gv.values();
    ASSERT_TRUE(std::is_sorted(partition.begin(), partition.end()));
    arb_assert(partition.back() == values.size());
}
