#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest.h"

#include "../../src/communication/communicator.hpp"
#include "../../src/communication/global_policy.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // We need to set the communicator policy at the top level
    // this allows us to build multiple communicators in the tests
    nest::mc::communication::global_policy_guard global_guard(argc, argv);

    return RUN_ALL_TESTS();
}
