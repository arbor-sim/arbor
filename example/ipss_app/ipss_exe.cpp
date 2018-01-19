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

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -n, --count=int        (10000)  Number of individual Poisson cell to run.\n"
"  -b, --begin=float      (0.0)    Start time (in ms) when to start generating\n"
"  -e, --end=float        (1000.0) End time (in ms) when to end generation\n"
"  -s, --sample=float     (0.1)    Internal sample rate for Poisson process\n"
"  -i, --interpolate=0/1  (1)   Interpolate between the supplied time-rate pairs\n"
"\n"
"  --pairs=path           (option)     Path to file with 'float, float' time rate pairs\n"
"  --output=path          (./spikes.gdf) Export produced spikes to this path\n"

"  -h, --help             Emit this message and exit.\n"
"\n"
"Create a group of Inhomogeneous Poisson Spike Sources\n"
"Run the cells and output the produced spikes to file\n"
"\n"
"Output of the default settings can be used in parse_and_plot.py to generate\n"
"the example plot in the how-to at 'http://arbor.readthedocs.io/en/latest/'\n"
"\n"
" Default time varying inhomogeneous spike rate:\n"
" hz.|                     \n"
" 240|     _-_             \n"
"    |    -   -  -         \n"
"    |   -     -- -        \n"
" 0  |__-__________-__     \n"
"      100        900   ms \n"
;

int main(int argc, char** argv) {
    // options
    unsigned n_cells = 10000;
    arb::time_type begin = 0.0;
    arb::time_type end = 1000.0;
    arb::time_type sample_delta = 0.1;
    optional<std::string> time_rate_path;
    std::string output_path = "spikes.gdf";
    bool interpolate = true;

    try {
        auto arg = argv + 1;
        while (*arg) {
            if (auto o = to::parse_opt<int>(arg, 'n', "count")) {
                n_cells = *o;
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
            else if (auto o = to::parse_opt(arg, 'h', "help")) { to::usage(argv[0], usage_str);
                return 0;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
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

    // Get the rate vector from file or use the default
    std::vector<std::pair<arb::time_type, double>> time_rate_pairs;
    if (time_rate_path) {
        time_rate_pairs = ipss_impl::parse_time_rate_from_path(time_rate_path.value());
    }
    else {
        time_rate_pairs = ipss_impl::default_time_rate_pairs();
    }

    // Create, and run the cells, get the spikes
    std::vector<arb::spike> produced_spikes = ipss_impl::create_and_run_ipss_cell_group(
        n_cells, begin, end, sample_delta, time_rate_pairs, interpolate);

    // Output the spikes to file
    ipss_impl::write_spikes_to_path(produced_spikes, output_path);

    return 0;
}
