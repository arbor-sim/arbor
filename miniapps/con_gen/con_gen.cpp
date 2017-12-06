#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <utility>

#include <tinyopt.hpp>
#include <util/optional.hpp>
#include <common_types.hpp>

#include "connection_generator.hpp"
#include "con_gen_utils.hpp"


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
    optional<std::string> population_cfg_path;
    optional<std::string> projection_cfg_path;

    optional<std::string> gid_path;
    std::string output_path = "./synapses.dat";

    try {
        auto arg = argv + 1;
        while (*arg) {
            if (auto o = to::parse_opt<std::string>(arg, 0, "populations")) {
                population_cfg_path = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "projections")) {
                projection_cfg_path = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "gids")) {
                gid_path = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "output")) {
                output_path = *o;
            }
            else if (auto o = to::parse_opt(arg, 'h', "help")) {
                to::usage(argv[0], usage_str);
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

    // If we have a supplied populations or connectome both should be supplied
    if (population_cfg_path || projection_cfg_path) {
        if (!population_cfg_path && projection_cfg_path) {
            throw con_gen_util::con_gen_error("population_cfg_path or projection_cfg_path alone is not valid.");
        }
    }

    // Parse the populations from file or use the default
    std::vector<arb::population> populations;
    if (population_cfg_path) {
        populations = con_gen_util::parse_populations_from_path(population_cfg_path.get());
    }
    else {
        populations = con_gen_util::default_populations();
    }


    // Parse the connectome from file or use the default
    std::vector<arb::projection>  connectome;
    if (projection_cfg_path) {
        connectome = con_gen_util::parse_projections_from_path(projection_cfg_path.get());
    }
    else {
        connectome = con_gen_util::default_connectome();
    }

    // gids we want to poll
    std::vector<arb::cell_gid_type> gids;
    if (gid_path) {
        gids = con_gen_util::parse_gids_from_path(gid_path.get());
    }
    else {
        gids = { 10320, 12003, 17997, 19580,
            15070, 5030,  // These two are shifted !!
            320, 2003, 7997, 9580, 5500 };
    }


    // The connection generator
    arb::connection_generator gen(populations, connectome);

    std::ofstream outfile(output_path);

    if (outfile) {
        // Pick some neurons on the borders to see correct periodic boundary behaviour
        for (auto gid : gids)
        {
            auto synapses = gen.synapses_on(gid);
            for (auto synapse : synapses) {
                outfile << synapse.gid << "," << synapse.weight << "," << synapse.delay << "\n";
            }
        }
    }
    else {
        throw con_gen_util::con_gen_error("Could not open supplied output_path");
    }

    return 0;
}
