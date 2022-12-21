#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <arbor/context.hpp>

#include <sup/ioutil.hpp>
#include <tinyopt/tinyopt.h>

#ifdef TEST_MPI
#include <arborenv/with_mpi.hpp>
#endif

#include "distributed_context.hpp"
#include "execution_context.hpp"

#include "distributed_listener.hpp"

#include "test.hpp"

using namespace arb;

context g_context = make_context();

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -d, --dryrun        Number of dry run ranks\n"
"  -h, --help          Display usage information and exit\n";

int main(int argc, char **argv) {
    proc_allocation alloc;
    alloc.gpu_id = -1;

#ifdef TEST_MPI
    arbenv::with_mpi guard(argc, argv, false);
    g_context = arb::make_context(alloc, MPI_COMM_WORLD);
#elif defined(TEST_LOCAL)
    g_context = arb::make_context(alloc);
#else
#error "define TEST_MPI or TEST_LOCAL for distributed test"
#endif

    // initialize google test environment
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    // replace original printer with our custom printer
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new distributed_listener("run_"+g_context->distributed->name(), g_context));

    int return_value = 0;
    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto comm_size = to::parse<unsigned>(arg, "-d", "--dryrun")) {
                if (*comm_size==0) {
                    throw to::user_option_error("number of dry run ranks must be positive");
                }
                // Note that this must be set again for each test that uses a different
                // number of cells per domain, e.g.
                //      policy::set_sizes(policy::size(), new_cells_per_rank)
                // TODO: fix when dry run mode reimplemented
                //policy::set_sizes(*comm_size, 0);
            }
            else if (to::parse(arg, "-h", "--help")) {
                to::usage(argv[0], usage_str);
                return 0;
            }
            else {
                throw to::option_error("unrecognized option", *arg);
            }
        }

        // record the local return value for tests run on this mpi rank
        //      0 : success
        //      1 : failure
        return_value = RUN_ALL_TESTS();
    }
    catch (to::option_error& e) {
        to::usage_error(argv[0], usage_str, e.what());
        return_value = 1;
    }
    catch (std::exception& e) {
        std::cout << "caught exception: " << e.what() << std::endl;
        return_value = 1;
    }

    // perform global collective, to ensure that all ranks return
    // the same exit code
    return g_context->distributed->max(return_value);
}
