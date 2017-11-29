#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <utility>

#include <tinyopt.hpp>
#include <util/optional.hpp>
#include <common_types.hpp>
#include <spike.hpp>

#include "ipss_util.hpp"

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
    unsigned n_cells = 10000;
    arb::time_type begin = 0.0;
    arb::time_type end = 1000.0;
    arb::time_type sample_delta = 0.1;
    optional<unsigned> rng_seed;
    optional<std::string> time_rate_path;
    std::string output_path = "spikes.gdf";
    bool interpolate = true;

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
            else if (auto o = to::parse_opt<arb::time_type>(arg, 's', "sample")) {
                sample_delta = *o;
            }
            else if (auto o = to::parse_opt<bool>(arg, 'i', "interpolate")) {
                interpolate = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "pairs")) {
                time_rate_path = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "output")) {
                output_path = *o;
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

        // Get the rate vector from file or default
        std::vector<std::pair<arb::time_type, double>> time_rate_pairs;
        if (time_rate_path) {
            time_rate_pairs = ipss_impl::parse_time_rate_from_path(time_rate_path.get());
        }
        else {

            time_rate_pairs = ipss_impl::default_time_rate_pairs();
        }

        // Run the cells
        std::vector<arb::spike> produced_spikes = ipss_impl::create_and_run_ipss_cell_group(
            n_cells, begin, end, sample_delta, time_rate_pairs, interpolate
        );

        // Output the spikes to file
        ipss_impl::write_spikes_to_path(produced_spikes, output_path);
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

