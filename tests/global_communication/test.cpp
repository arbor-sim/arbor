#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include "../gtest.h"

#include "mpi_listener.hpp"

#include <tinyopt.hpp>
#include <communication/communicator.hpp>
#include <communication/distributed_context.hpp>
#include <util/ioutil.hpp>


using namespace arb;

distributed_context g_context;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -d, --dryrun        Number of dry run ranks\n"
"  -h, --help          Display usage information and exit\n";

int main(int argc, char **argv) {

    // We need to set the communicator policy at the top level
    // this allows us to build multiple communicators in the tests
    #ifdef ARB_HAVE_MPI
    mpi::scoped_guard guard(&argc, &argv);
    g_context = mpi_context(MPI_COMM_WORLD);
    #endif

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_global_communication", &g_context));

    int return_value = 0;
    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto comm_size = to::parse_opt<unsigned>(arg, 'd', "dryrun")) {
                if (*comm_size==0) {
                    throw to::parse_opt_error(*arg, "must be positive integer");
                }
                // Note that this must be set again for each test that uses a different
                // number of cells per domain, e.g.
                //      policy::set_sizes(policy::size(), new_cells_per_rank)
                // TODO: fix when dry run mode reimplemented
                //policy::set_sizes(*comm_size, 0);
            }
            else if (auto o = to::parse_opt(arg, 'h', "help")) {
                to::usage(argv[0], usage_str);
                return 0;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        // record the local return value for tests run on this mpi rank
        //      0 : success
        //      1 : failure
        return_value = RUN_ALL_TESTS();
    }

    catch (to::parse_opt_error& e) {
        to::usage(argv[0], usage_str, e.what());
        return_value = 1;
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        return_value = 1;
    }

    // perform global collective, to ensure that all ranks return
    // the same exit code
    return g_context.max(return_value);
}
