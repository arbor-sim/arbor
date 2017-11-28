#include <iostream>
#include <fstream>
#include <random>

#include <tinyopt.hpp>
#include <util/optional.hpp>
#include <common_types.hpp>

#include "ipss_util.hpp"

#include <ipss_cell_description.hpp>
#include <ipss_cell_group.hpp>

#include "../simple_recipes.hpp"



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
    unsigned n_cells = 1;
    arb::time_type begin = 0.0;
    arb::time_type end = 10.0;
    optional<unsigned> rng_seed;
    optional<std::string> time_rate_path;


    try {
        auto arg = argv + 1;
        while (*arg) {
            if (auto o = to::parse_opt<int>(arg, 'n', "count")) {
                n_cells = *o;
            }
            else if (auto o = to::parse_opt<unsigned>(arg, 's', "seed")) {
                rng_seed = *o;
            }
            else if (auto o = to::parse_opt<arb::time_type>(arg, 'b', "begin")) {
                begin = *o;
            }
            else if (auto o = to::parse_opt<arb::time_type>(arg, 'e', "end")) {
                end = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "pairs")) {
                time_rate_path = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "output")) {
                time_rate_path = *o;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        std::minstd_rand g;
        if (rng_seed) {
            g.seed(rng_seed.get());
            std::cout << "seed" << rng_seed.get() << std::endl;

        }

        std::vector<std::pair<arb::time_type, double>> time_rate_pairs;
        if (time_rate_path) {
            time_rate_pairs = arb::parse_time_pair_in_path(time_rate_path.get());
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

