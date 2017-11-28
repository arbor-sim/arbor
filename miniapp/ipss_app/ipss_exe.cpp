#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

#include <iostream>

#include <tinyopt.hpp>
#include <util/optional.hpp>

namespace to = arb::to;
using arb::util::optional;
using arb::util::nothing;
using arb::util::just;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -n, --count=N      Number of morphologies to generate.\n"
"  -s, --seed=seed    Use L-system MODEL for generation (see below).\n"
"  --csv=FILE         Output morphologies as SWC to FILE (see below).\n"
"  -h, --help         Emit this message and exit.\n"
"\n"
"Generate artificial neuron morphologies based on L-system descriptions.\n"
"\n"
"If a FILE arrgument contains a '%', then one file will be written for\n"
"each generated morphology, with the '%' replaced by the index of the\n"
;

int main(int argc, char** argv) {
    // options
    int n_cells = 1;
    optional<unsigned> rng_seed;
    optional<std::string> time_rate_csv;


    try {
        auto arg = argv + 1;
        while (*arg) {
            if (auto o = to::parse_opt<int>(arg, 'n', "count")) {
                n_cells = *o;
            }
            if (auto o = to::parse_opt<unsigned>(arg, 's', "seed")) {
                rng_seed = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "csv")) {
                time_rate_csv = *o;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        std::minstd_rand g;
        if (rng_seed) {
            g.seed(rng_seed.get());
        }

        if (time_rate_csv) {

            std::cout << "File with time rate pairs:" << time_rate_csv.get() << std::endl;
        }



    }
    catch (to::parse_opt_error& e) {
        std::cerr << argv[0] << ": " << e.what() << "\n";
        std::cerr << "Try '" << argv[0] << " --help' for more information.\n";
        std::exit(2);
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}

