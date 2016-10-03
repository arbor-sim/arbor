#include "gtest.h"

#include <algorithm>
#include <vector>

#include "../include/HostCoordinator.hpp"
#include "../include/SplitRange.hpp"
#include "../include/Vector.hpp"

TEST(SplitRange, split) {
    using namespace memory;

    using vector_type = HostVector<float>;
    using view_type = vector_type::view_type;

    const size_t num_splits=13;
    const size_t range_length=100;

    vector_type vector(range_length);

    std::vector<view_type> splits;
    std::vector<Range> ranges;

    for(auto s : SplitRange(Range(0,range_length), num_splits)) {
        splits.push_back(vector(s));
        ranges.push_back(s);
    }

    EXPECT_EQ(splits.size(), num_splits);
    EXPECT_EQ(ranges.size(), num_splits);
    for(int i=0; i<num_splits; ++i)
        EXPECT_EQ(splits[i].size(), ranges[i].size());
}
