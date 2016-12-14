#include "../gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>

using namespace nest::mc;

using time_type = float;
using communicator_type = communication::communicator<time_type, communication::global_policy>;

TEST(communicator, setup) {
    /*
    using policy = communication::global_policy;

    auto num_domains = policy::size();
    auto rank = policy::id();

    auto counts = policy.gather_all(1);
    EXPECT_EQ(counts.size(), unsigned(num_domains));
    for(auto i : counts) {
        EXPECT_EQ(i, 1);
    }

    auto part = util::parition_view(counts);
    for(auto p : part) {
        EXPECT_EQ(p.second-p.first, 1);
    }
    */
}
