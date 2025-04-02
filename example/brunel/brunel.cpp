#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <vector>

#include <tinyopt/tinyopt.h>

#include <arbor/context.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/version.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborenv/default_env.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <arborenv/with_mpi.hpp>
#endif

using namespace arb;

// Holds the options for a simulation run.
// Default constructor gives default options.
struct cl_options {
    // Cell parameters:
    uint32_t nexc = 400;
    uint32_t ninh = 100;
    uint32_t next = 40;
    double syn_per_cell_prop = 0.05;
    float weight = 1.2;
    float delay = 0.1;
    float rel_inh_strength = 1;
    double poiss_lambda = 1;
    // use cable cells instead of LIF
    bool use_cc = false;
    // Simulation running parameters:
    double tfinal = 100.;
    double dt = 0.05;
    uint32_t group_size = 10;
    uint32_t seed = 42;
    // Parameters for spike output.
    std::string spike_file_output = "";
    // Be more verbose with informational messages.
    bool verbose = false;
};

std::ostream& operator<<(std::ostream& o, const cl_options& opt);

std::optional<cl_options> read_options(int argc, char** argv);

void banner(context ctx);

// Add m unique connection from gids in interval [start, end) - gid.
void add_subset(cell_gid_type gid,
                cell_gid_type start, cell_gid_type end,
                unsigned m,
                const std::string& src, const std::string& tgt,
                float weight, float delay,
                std::vector<cell_connection>& conns);

/*
   A Brunel network consists of nexc excitatory LIF neurons and ninh inhibitory
   LIF neurons. Each neuron in the network receives in_degree_prop * nexc
   excitatory connections chosen randomly, in_degree_prop * ninh inhibitory
   connections and next (external) Poisson connections. All the connections have
   the same delay. The strenght of excitatory and Poisson connections is given
   by parameter weight, whereas the strength of inhibitory connections is
   rel_inh_strength * weight. Poisson neurons all spike independently with
   expected number of spikes given by parameter poiss_lambda. Because of the
   refractory period, the activity is mostly driven by Poisson neurons and
   recurrent connections have a small effect.
 */
class brunel_recipe: public recipe {
public:
    brunel_recipe(cell_size_type nexc,
                  cell_size_type ninh,
                  cell_size_type next,
                  double in_degree_prop,
                  float weight,
                  float delay,
                  float rel_inh_strength,
                  double poiss_lambda,
                  int seed=42,
                  bool use_cc=false):
        ncells_exc_(nexc), ncells_inh_(ninh), delay_(delay), seed_(seed), use_cable_cells(use_cc) {
        // Make sure that in_degree_prop in the interval (0, 1]
        if (in_degree_prop <= 0.0 || in_degree_prop > 1.0) {
            throw std::domain_error("The proportion of incoming connections should be in the interval (0, 1].");
        }

        // Set up the parameters.
        weight_exc_ = weight;
        weight_inh_ = -rel_inh_strength * weight_exc_;
        weight_ext_ =  weight;
        in_degree_exc_ = std::round(in_degree_prop * nexc);
        in_degree_inh_ = std::round(in_degree_prop * ninh);
        // each cell receives next incoming Poisson sources with mean rate poiss_lambda, which is equivalent
        // to a single Poisson source with mean rate next*poiss_lambda
        lambda_ = next * poiss_lambda;
        // construct cable cell prototype
        {
            arb::segment_tree tree;
            tree.append(arb::mnpos, {-1.0, 0, 0, 1.0}, {1, 0, 0, 1.0}, 1);

            auto dec = arb::decor{}
                .paint(arb::reg::tagged(1), arb::density("hh"))
                .place(arb::ls::location(0, 0.5), arb::synapse("expsyn"), "tgt")
                .place(arb::ls::location(0, 0.5), arb::threshold_detector(-10 * U::mV), "src");

            cable = arb::cable_cell(tree, dec, {});
        }
        prop.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return ncells_exc_ + ncells_inh_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        if (use_cable_cells) {
            return cell_kind::cable;
        }
        else {
            return cell_kind::lif;
        }
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        // Add incoming excitatory and inhibitory connections.
        add_subset(gid, 0,           ncells_exc_,               in_degree_exc_, "src", "tgt", weight_exc_, delay_, connections);
        add_subset(gid, ncells_exc_, ncells_inh_ + ncells_exc_, in_degree_inh_, "src", "tgt", weight_inh_, delay_, connections);
        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (use_cable_cells) {
            return cable;
        }
        else {
            return lif;
        }
    }

    std::any get_global_properties(cell_kind) const override { return prop; }

    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        return {poisson_generator({"tgt"}, weight_ext_, 0*arb::units::ms, lambda_*arb::units::kHz, gid + seed_)};
    }

private:
    // Number of excitatory cells.
    cell_size_type ncells_exc_;
    // Number of inhibitory cells.
    cell_size_type ncells_inh_;
    // Weight of excitatory synapses.
    float weight_exc_;
    // Weight of inhibitory synapses.
    float weight_inh_;
    // Weight of external Poisson cell synapses.
    float weight_ext_;
    // Delay of all synapses.
    float delay_;
    // Number of connections that each neuron receives from excitatory population.
    int in_degree_exc_;
    // Number of connections that each neuron receives from inhibitory population.
    int in_degree_inh_;
    // Expected number of poisson spikes.
    double lambda_;
    // Seed used for the Poisson spikes generation.
    int seed_;

    // LIF cell prototype
    arb::lif_cell lif = {
        .source="src",
        .target="tgt",
        .tau_m = 10*U::ms,
        .V_th = 10*U::mV,
        .C_m = 20*U::pF,
        .E_L = 0*U::mV,
        .V_m = 0*U::mV,
        .t_ref = 2*U::ms,
    };
    // Cable cell prototype
    arb::cable_cell cable;
    arb::cable_cell_global_properties prop;
    bool use_cable_cells = true;
};

int main(int argc, char** argv) {
    bool root = true;

    try {
#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
        unsigned num_threads = arbenv::default_concurrency();
        int gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(arb::proc_allocation{num_threads, gpu_id}, MPI_COMM_WORLD);
        root = arb::rank(context)==0;
#else
        auto context = arb::make_context(arbenv::default_allocation());
#endif

#ifdef ARB_PROFILE_ENABLED
        arb::profile::profiler_initialize(context);
#endif

        std::cout << sup::mask_stream(root);
        banner(context);

        arb::profile::meter_manager meters;
        meters.start(context);

        // read parameters
        auto o = read_options(argc, argv);
        if (!o) {return 0; }
        cl_options options = o.value();

        std::fstream spike_out;
        auto spike_file_output = options.spike_file_output;
        if (spike_file_output != "" && root) {
            spike_out = sup::open_or_throw(spike_file_output, std::ios_base::out, false);
        }

        meters.checkpoint("setup", context);

        // The size of excitatory population.
        cell_size_type nexc = options.nexc;

        // The size of inhibitory population.
        cell_size_type ninh = options.ninh;

        // The size of Poisson (external) population.
        cell_size_type next = options.next;

        // Fraction of connections each neuron receives from each of the 3 populations.
        double in_degree_prop = options.syn_per_cell_prop;

        // Weight of excitatory and poisson connections.
        float w = options.weight;

        // Delay of all the connections.
        float d = options.delay;

        // Relative strength of inhibitory connections with respect to excitatory connections.
        float rel_inh_strength = options.rel_inh_strength;

        // Expected number of spikes from a single poisson cell per ms.
        int poiss_lambda = options.poiss_lambda;

        // The number of cells in a single cell group.
        cell_size_type group_size = options.group_size;

        unsigned seed = options.seed;

        brunel_recipe recipe(nexc, ninh, next, in_degree_prop, w, d, rel_inh_strength, poiss_lambda, seed, options.use_cc);

        partition_hint_map hints;
        hints[cell_kind::lif].cpu_group_size = group_size;
        hints[cell_kind::cable].cpu_group_size = group_size;
        hints[cell_kind::lif].gpu_group_size = group_size;
        hints[cell_kind::cable].gpu_group_size = group_size;
        auto dec = partition_load_balance(recipe, context, hints);

        simulation sim(recipe,
                       context,
                       dec);

        // Set up spike recording.
        std::vector<arb::spike> recorded_spikes;
        if (spike_out) {
            sim.set_global_spike_callback([&recorded_spikes](auto& spikes) {
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        }

        meters.checkpoint("model-init", context);

        // Run simulation.
        sim.run(options.tfinal*arb::units::ms, options.dt*arb::units::ms);

        meters.checkpoint("model-simulate", context);

        // Output spikes if requested.
        if (spike_out) {
            spike_out << std::fixed << std::setprecision(4);
            for (auto& s: recorded_spikes) {
                spike_out << s.source.gid << ' '
                          << s.time << '\n';
            }
        }

        // output profile and diagnostic feedback
        std::cout << profile::profiler_summary() << "\n"
                  << "\nThere were " << sim.num_spikes() << " spikes\n";

        auto report = profile::make_meter_report(meters, context);
        std::cout << report;
        if (root) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << sup::to_json(report) << "\n";
        }
    }
    catch (std::exception& e) {
        // only print errors on master
        std::cerr << sup::mask_stream(root)
                  << e.what() << "\n";
        return 1;
    }
    return 0;
}

void banner(context ctx) {
    std::cout << "==========================================\n";
    std::cout << "  Brunel model miniapp\n";
    std::cout << "  - distributed : " << arb::num_ranks(ctx)
              << (arb::has_mpi(ctx)? " (mpi)": " (serial)") << "\n";
    std::cout << "  - threads     : " << arb::num_threads(ctx) << "\n";
    std::cout << "  - gpus        : " << (arb::has_gpu(ctx)? "yes": "no") << "\n";
    std::cout << "==========================================\n";
}

// simple, compiler independent int in range
template<typename T, typename G>
T rand_range(G& gen, T lo, T hi) { return lo + gen() * double(hi - lo) / double(G::max() - G::min()); }

void add_subset(cell_gid_type gid,
                cell_gid_type start, cell_gid_type end,
                unsigned m,
                const std::string& src, const std::string& tgt,
                float weight, float delay,
                std::vector<cell_connection>& conns) {
    // We can only add this many connections!
    auto gid_in_range = int(gid >= start && gid < end);
    if (m + start + gid_in_range >= end) throw std::runtime_error("Requested too many connections from the given range of gids.");
    // Exclude ourself
    std::set<cell_gid_type> seen{gid};
    std::mt19937 gen(gid + 42);
    while(m > 0) {
        cell_gid_type val = rand_range(gen, start, end);
        if (!seen.count(val)) {
            conns.push_back({{val, src}, {tgt}, weight, delay*U::ms});
            seen.insert(val);
            m--;
        }
    }
}


// Read options from (optional) json file and command line arguments.
std::optional<cl_options> read_options(int argc, char** argv) {
    using namespace to;
    auto usage_str = "\n"
                     "-n|--n-excitatory      [Number of cells in the excitatory population]\n"
                     "-m|--n-inhibitory      [Number of cells in the inhibitory population]\n"
                     "-e|--n-external        [Number of incoming Poisson (external) connections per cell]\n"
                     "-p|--in-degree-prop    [Proportion of the connections received per cell]\n"
                     "-w|--weight            [Weight of excitatory connections]\n"
                     "-d|--delay             [Delay of all connections]\n"
                     "-g|--rel-inh-w         [Relative strength of inhibitory synapses with respect to the excitatory ones]\n"
                     "-l|--lambda            [Mean firing rate from a single poisson cell (kHz)]\n"
                     "-t|--tfinal            [Length of the simulation period (ms)]\n"
                     "-s|--dt                [Simulation time step (ms)]\n"
                     "-G|--group-size        [Number of cells per cell group]\n"
                     "-S|--seed              [Seed for poisson spike generators]\n"
                     "-f|--write-spikes      [Save spikes to file]\n"
                     "-c|--use-cable-cells   [Use a cable cell model]\n"
                     "-v|--verbose           [Print more verbose information to stdout]\n";

    cl_options opt;
    auto help = [argv0 = argv[0], &usage_str] {
        to::usage(argv0, usage_str);
    };

    to::option options[] = {
            { opt.nexc,                        "-n", "--n-excitatory" },
            { opt.ninh,                        "-m", "--n-inhibitory" },
            { opt.next,                        "-e", "--n-external" },
            { opt.syn_per_cell_prop,           "-p", "--in-degree-prop" },
            { opt.weight,                      "-w", "--weight" },
            { opt.delay,                       "-d", "--delay" },
            { opt.rel_inh_strength,            "-g", "--rel-inh-w" },
            { opt.poiss_lambda,                "-l", "--lambda" },
            { opt.tfinal,                      "-t", "--tfinal" },
            { opt.dt,                          "-s", "--dt" },
            { opt.group_size,                  "-G", "--group-size" },
            { opt.seed,                        "-S", "--seed" },
            { opt.spike_file_output,           "-f", "--write-spikes" },
            { to::set(opt.use_cc),   to::flag, "-c", "--use-cable-cells" },
            { to::set(opt.verbose),  to::flag, "-v", "--verbose" },
            { to::action(help),      to::flag, to::exit, "-h", "--help" }
    };

    if (!to::run(options, argc, argv+1)) return {};
    if (argv[1]) throw to::option_error("unrecognized argument", argv[1]);

    if (opt.group_size < 1) {
        throw std::runtime_error("minimum of one cell per group");
    }

    if (opt.rel_inh_strength <= 0 || opt.rel_inh_strength > 1) {
        throw std::runtime_error("relative strength of inhibitory connections must be in the interval (0, 1].");
    }

    // If verbose output requested, emit option summary.
    if (opt.verbose) {
        std::cout << opt << "\n";
    }

    return opt;
}

std::ostream& operator<<(std::ostream& o, const cl_options& options) {
    o << "Simulation options:\n"
      << "  Excitatory cells                                           : " << options.nexc << "\n"
      << "  Inhibitory cells                                           : " << options.ninh << "\n"
      << "  Poisson connections per cell                               : " << options.next << "\n"
      << "  Proportion of synapses/cell from each population           : " << options.syn_per_cell_prop << "\n"
      << "  Weight of excitatory synapses                              : " << options.weight << "\n"
      << "  Relative strength of inhibitory synapses                   : " << options.rel_inh_strength << "\n"
      << "  Delay of all synapses                                      : " << options.delay << "\n"
      << "  Expected number of spikes from a single poisson cell per ms: " << options.poiss_lambda << "\n"
      << "\n"
      << "  Simulation time                                            : " << options.tfinal << "\n"
      << "  dt                                                         : " << options.dt << "\n"
      << "  Group size                                                 : " << options.group_size << "\n"
      << "  Seed                                                       : " << options.seed << "\n"
      << "  Spike file output                                          : " << options.spike_file_output << "\n";
    return o;
}
