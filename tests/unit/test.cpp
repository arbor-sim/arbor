#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include <communication/global_policy.hpp>

#include "../gtest.h"

int main(int argc, char **argv) {
    arb::communication::global_policy_guard g(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
