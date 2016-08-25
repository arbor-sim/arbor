#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest.h"

#include "mpi_listener.hpp"

#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <util/ioutil.hpp>

using namespace nest::mc;

int main(int argc, char **argv) {
    // We need to set the communicator policy at the top level
    // this allows us to build multiple communicators in the tests
    communication::global_policy_guard global_guard(argc, argv);

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_global_communication"));

    // record the local return value for tests run on this mpi rank
    //      0 : success
    //      1 : failure
    auto result = RUN_ALL_TESTS();

    // perform global collective, to ensure that all ranks return
    // the same exit code
    return communication::global_policy::max(result);
}
