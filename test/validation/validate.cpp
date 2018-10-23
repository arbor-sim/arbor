#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <exception>

#include <sup/tinyopt.hpp>

#include "../gtest.h"

#include "validation_data.hpp"

using namespace arb;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -v, --verbose       Print results to stdout\n"
"  -o, --output=FILE   Save traces from simulations to FILE\n"
"  -p, --path=DIR      Look for validation reference data in DIR\n"
"  -m, --max-comp=N    Run convergence tests to a maximum of N\n"
"                      compartments per segment\n"
"  -d, --min-dt=DT     Run convergence tests with a minimum timestep DT [ms]\n"
"  -s, --sample-dt=DT  Sample rate for simulations [ms]\n"
"  -h, --help          Display usage information and exit\n"
"\n"
"Validation data is searched by default in the directory specified by\n"
"ARB_DATADIR at compile time. If ARB_DATADIR does not correspond to a\n"
"directory, the tests will try the paths './validation/data' and\n"
"'../validation/data'. This default path can be overridden with the\n"
"ARB_DATADIR environment variable, or with the -p command-line option.\n";

int main(int argc, char **argv) {
    using to::parse_opt;

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
            else if (auto o = parse_opt<float>(arg, 'd', "min-dt")) {
                g_trace_io.set_min_dt(*o);
            }
            else if (auto o = parse_opt<float>(arg, 's', "sample-dt")) {
                g_trace_io.set_sample_dt(*o);
            }
            else if (auto o = parse_opt(arg, 'v', "verbose")) {
                g_trace_io.set_verbose(true);
            }
            else if (auto o = parse_opt(arg, 'h', "help")) {
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
