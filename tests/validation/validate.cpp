#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <exception>

#include "gtest.h"

#include <communication/global_policy.hpp>

#include "tinyopt.hpp"
#include "validation_data.hpp"

using namespace nest::mc;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -v, --verbose       Print results to stdout\n"
"  -o, --output=FILE   Save traces from simulations to FILE\n"
"  -p, --path=DIR      Look for validation reference data in DIR\n"
"  -m, --max-comp=N    Run convergence tests to a maximum of N\n"
"                      compartments per segment\n"
"  -h, --help          Display usage information and exit\n";

int main(int argc, char **argv) {
    using to::parse_opt;

    communication::global_policy_guard global_guard(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto o = parse_opt<std::string>(arg, 'p', "path")) {
                g_trace_io.set_datadir(*o);
            }
            else if (auto o = parse_opt<std::string>(arg, 'o', "output")) {
                g_trace_io.set_output(*o);
            }
            else if (auto o = parse_opt<int>(arg, 'm', "max-comp")) {
                g_trace_io.set_max_ncomp(*o);
            }
            else if (auto o = parse_opt<void>(arg, 'v', "verbose")) {
                g_trace_io.set_verbose(true);
            }
            else if (auto o = parse_opt<void>(arg, 'h', "help")) {
                to::usage(argv[0], usage_str);
                return 0;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        return RUN_ALL_TESTS();
    }

    catch (to::parse_opt_error& e) {
        to::usage(argv[0], usage_str, e.what());
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
