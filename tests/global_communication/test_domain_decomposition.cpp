#include "../gtest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>

using namespace nest::mc;

using communicator_type = communication::communicator<communication::global_policy>;

inline bool is_dry_run() {
    return communication::global_policy::kind() ==
        communication::global_policy_kind::dryrun;
}

TEST(domain_decomp, basic) {
/*
    using policy = communication::global_policy;

    const auto num_domains = policy::size();
    const auto rank = policy::id();
*/


}
