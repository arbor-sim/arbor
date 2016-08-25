#include "gtest.h"

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
}
