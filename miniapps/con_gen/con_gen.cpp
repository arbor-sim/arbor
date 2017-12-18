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
" A small validation program for validating the generation of synaptic connection \n"
" Between 2d sheets of cells on a grid, with optionally a periodic border. Synapses are output\n"
" to file and have a configurable weight and delay distribution. The dimensions of the \n "
" populations can have arbitrary dimensions, although projecting on a population\n"
" With a different side ratio is *not* tested\n"
"\n"
" The default settings of this app with use a default projection between two\n"
" 100 * 100 populations with periodic borders. and output the synapse to synapses.dat\n"
" This output can be parsed with the default parse_and_plot.py python script \n"
" When using your own population you have to supply this with commandline of the python script !! \n"
"\n"
"\n"
"   [OPTION] \n"
"  --populations=path     path with config for populations  \n"
"  --projections=path     path with config for projections between the populations\n"
"       (These two path must both be set or not set at all)\n "
"  --gids=path            path with config for gids we want the synapse to output for\n"
"  --output=path          output spikes to this path\n"
"\n"
"   When an error of parsing of config files occurs the app will fail silently (or loudly)! \n"
"\n"
"   *** configuration file syntax  ***\n"
"   * population config: \n"
"   The lines are parsed, as comma separated values: \n"
"   cell_on_side_x, cell_on_side_y, periodic  \n"
"   with types:\n"
"   unsigned, unsigned, 0 -or- 1 \n"
"\n"
"   * projection config: \n"
"   The lines are parsed, as comma separated values: \n"
"   idx_pre_polation, idx_post_polation, count, sd\n"
"   mean_weight, weight_sd, min_delay, delay_per_sd\n"
"  with types:\n"
"   unsigned, unsigned, unsigned, float, float, float, float, float\n"
"\n"
"   * gids config: \n"
"   The lines are parsed, as comma separated values: \n"
"   IF a line starts with a comma, it is parsed as a comma separated list of gids\n"
"   finished with a '<' character\n"
"   If a line starts with a - (or any other character). It is parsed as two\n"
"   comma separated gids and assumed to be a range\n"
"   Types of parsed numbers is unsigned\n"
"\n"
"  *** projection parameters explanation: ***\n"
"  - sd             sd of the normal distributed used to sample the pre_synaptic\n"
"                   The dimensions of the pre-population is sampled as if it has size 1.0 * 1.0\n"
"                   (ration of x and y is accounted for.)\n"
"  - count          Number of synapse to makee. When sampling from a non periodic population\n"
"                   this count can be lower (akin with a sample in-vitro) \n"
" - weight_mean     Mean synaptic weight for the created synapse\n"
"  - weight_sd      Standard deviation around mean for sampling the weights\n"
"  - delay_min      Minimal delay of the created synapse\n"
"  - delay_per_sd   Delay increase by sd distance between neurons\n"
"\n"
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
    std::vector<arb_con_gen::population> populations;
    if (population_cfg_path) {
        populations = con_gen_util::parse_populations_from_path(population_cfg_path.get());
    }
    else {
        populations = con_gen_util::default_populations();
    }


    // Parse the connectome from file or use the default
    std::vector<arb_con_gen::projection>  connectome;
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
    arb_con_gen::connection_generator gen(populations, connectome);

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
