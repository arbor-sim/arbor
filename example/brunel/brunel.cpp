#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <tinyopt/smolopt.h>

#include <arbor/context.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/util/optional.hpp>
#include <arbor/version.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <sup/ioutil.hpp>
#include <sup/json_meter.hpp>
#include <sup/path.hpp>
#include <sup/strsub.hpp>

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

    // Simulation running parameters:
    double tfinal = 100.;
    double dt = 1;
    uint32_t group_size = 10;
    uint32_t seed = 42;

    // Parameters for spike output.
    bool spike_file_output = false;

    // Turn on/off profiling output for all ranks.
    bool profile_only_zero = false;

    // Be more verbose with informational messages.
    bool verbose = false;
};

std::ostream& operator<<(std::ostream& o, const cl_options& opt);

util::optional<cl_options> read_options(int argc, char** argv);

void banner(const context& ctx);

// Samples m unique values in interval [start, end) - gid.
// We exclude gid because we don't want self-loops.
std::vector<cell_gid_type> sample_subset(cell_gid_type gid, cell_gid_type start, cell_gid_type end,  unsigned m);

/*
   A Brunel network consists of nexc excitatory LIF neurons and ninh inhibitory LIF neurons.
   Each neuron in the network receives in_degree_prop * nexc excitatory connections
   chosen randomly, in_degree_prop * ninh inhibitory connections and next (external) Poisson connections.
   All the connections have the same delay. The strenght of excitatory and Poisson connections is given by
   parameter weight, whereas the strength of inhibitory connections is rel_inh_strength * weight.
   Poisson neurons all spike independently with expected number of spikes given by parameter poiss_lambda.
   Because of the refractory period, the activity is mostly driven by Poisson neurons and
   recurrent connections have a small effect.
 */
class brunel_recipe: public recipe {
public:
    brunel_recipe(cell_size_type nexc, cell_size_type ninh, cell_size_type next, double in_degree_prop,
                  float weight, float delay, float rel_inh_strength, double poiss_lambda, int seed = 42):
        ncells_exc_(nexc), ncells_inh_(ninh), delay_(delay), seed_(seed) {
        // Make sure that in_degree_prop in the interval (0, 1]
        if (in_degree_prop <= 0.0 || in_degree_prop > 1.0) {
            std::out_of_range("The proportion of incoming connections should be in the interval (0, 1].");
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
    }

    cell_size_type num_cells() const override {
        return ncells_exc_ + ncells_inh_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        // Add incoming excitatory connections.
        for (auto i: sample_subset(gid, 0, ncells_exc_, in_degree_exc_)) {
            cell_member_type source{cell_gid_type(i), 0};
            cell_member_type target{gid, 0};
            cell_connection conn(source, target, weight_exc_, delay_);
            connections.push_back(conn);
        }

        // Add incoming inhibitory connections.
        for (auto i: sample_subset(gid, ncells_exc_, ncells_exc_ + ncells_inh_, in_degree_inh_)) {
            cell_member_type source{cell_gid_type(i), 0};
            cell_member_type target{gid, 0};
            cell_connection conn(source, target, weight_inh_, delay_);
            connections.push_back(conn);
        }
        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        auto cell = lif_cell();
        cell.tau_m = 10;
        cell.V_th = 10;
        cell.C_m = 20;
        cell.E_L = 0;
        cell.V_m = 0;
        cell.V_reset = 0;
        cell.t_ref = 2;
        return cell;
    }

    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;

        std::mt19937_64 G;
        G.seed(gid + seed_);

        time_type t0 = 0;
        cell_member_type target{gid, 0};

        gens.emplace_back(poisson_generator(target, weight_ext_, t0, lambda_, G));
        return gens;
    }

    cell_size_type num_sources(cell_gid_type) const override {
         return 1;
    }

    cell_size_type num_targets(cell_gid_type) const override {
        return 1;
    }

    cell_size_type num_probes(cell_gid_type) const override {
        return 0;
    }

    probe_info get_probe(cell_member_type probe_id) const override {
        return {};
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
};

int main(int argc, char** argv) {
    bool root = true;

    try {
        arb::proc_allocation resources;
        if (auto nt = arbenv::get_env_num_threads()) {
            resources.num_threads = nt;
        }
        else {
            resources.num_threads = arbenv::thread_concurrency();
        }

#ifdef ARB_MPI_ENABLED
        arbenv::with_mpi guard(argc, argv, false);
        resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(resources, MPI_COMM_WORLD);
        root = arb::rank(context)==0;
#else
        resources.gpu_id = arbenv::default_gpu();
        auto context = arb::make_context(resources);
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
        if (options.spike_file_output && root) {
            spike_out = sup::open_or_throw("./spikes.gdf", std::ios_base::out, false);
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

        brunel_recipe recipe(nexc, ninh, next, in_degree_prop, w, d, rel_inh_strength, poiss_lambda, seed);

        partition_hint_map hints;
        hints[cell_kind::lif].cpu_group_size = group_size;
        auto decomp = partition_load_balance(recipe, context, hints);

        simulation sim(recipe, decomp, context);

        // Set up spike recording.
        std::vector<arb::spike> recorded_spikes;
        if (spike_out) {
            sim.set_global_spike_callback([&recorded_spikes](auto& spikes) {
                    recorded_spikes.insert(recorded_spikes.end(), spikes.begin(), spikes.end());
                });
        }

        meters.checkpoint("model-init", context);

        // Run simulation.
        sim.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate", context);

        // Output spikes if requested.
        if (spike_out) {
            spike_out << std::fixed << std::setprecision(4);
            for (auto& s: recorded_spikes) {
                spike_out << s.source.gid << ' ' << s.time << '\n';
            }
        }

        // output profile and diagnostic feedback
        std::cout << profile::profiler_summary() << "\n";
        std::cout << "\nThere were " << sim.num_spikes() << " spikes\n";

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
        std::cerr << sup::mask_stream(root);
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

void banner(const context& ctx) {
    std::cout << "==========================================\n";
    std::cout << "  Brunel model miniapp\n";
    std::cout << "  - distributed : " << arb::num_ranks(ctx)
              << (arb::has_mpi(ctx)? " (mpi)": " (serial)") << "\n";
    std::cout << "  - threads     : " << arb::num_threads(ctx) << "\n";
    std::cout << "  - gpus        : " << (arb::has_gpu(ctx)? "yes": "no") << "\n";
    std::cout << "==========================================\n";
}

std::vector<cell_gid_type> sample_subset(cell_gid_type gid, cell_gid_type start, cell_gid_type end,  unsigned m) {
    std::set<cell_gid_type> s;
    std::mt19937 gen(gid + 42);
    std::uniform_int_distribution<cell_gid_type> dis(start, end - 1);
    while (s.size() < m) {
        auto val = dis(gen);
        if (val != gid) {
            s.insert(val);
        }
    }
    return {s.begin(), s.end()};
}

// Read options from (optional) json file and command line arguments.
util::optional<cl_options> read_options(int argc, char** argv) {
    using namespace to;
    auto usage_str = "\n"
                     "-n|--n-excitatory      [Number of cells in the excitatory population]\n"
                     "-m|--n-inhibitory      [Number of cells in the inhibitory population]\n"
                     "-e|--n-external        [Number of incoming Poisson (external) connections per cell]\n"
                     "-p|--in-degree-prop    [Proportion of the connections received per cell]\n"
                     "-w|--weight            [Weight of excitatory connections]\n"
                     "-d|--delay             [Delay of all connections]\n"
                     "-g|--rel-inh-w         [Relative strength of inhibitory synapses with respect to the excitatory ones]\n"
                     "-l|--lambda            [Expected number of spikes from a single poisson cell per ms]\n"
                     "-t|--tfinal            [Length of the simulation period (ms)]\n"
                     "-s|--dt                [Simulation time step (ms)]\n"
                     "-G|--group-size        [Number of cells per cell group]\n"
                     "-S|--seed              [Seed for poisson spike generators]\n"
                     "-f|--write-spikes      [Save spikes to file]\n"
                     "-z|--profile-rank-zero [Only output profile information for rank 0]\n"
                     "-v|--verbose           [Print more verbose information to stdout]\n";

    cl_options opt;
    auto help = [argv0 = argv[0], &usage_str] {
        to::usage(argv0, usage_str);
    };

    to::option options[] = {
            { opt.nexc,              "-n", "--n-excitatory" },
            { opt.ninh,              "-m", "--n-inhibitory" },
            { opt.next,              "-e", "--n-external" },
            { opt.syn_per_cell_prop, "-p", "--in-degree-prop" },
            { opt.weight,            "-w", "--weight" },
            { opt.delay,             "-d", "--delay" },
            { opt.rel_inh_strength,  "-g", "--rel-inh-w" },
            { opt.poiss_lambda,      "-l", "--lambda" },
            { opt.tfinal,            "-t", "--tfinal" },
            { opt.dt,                "-s", "--dt" },
            { opt.group_size,        "-G", "--group-size" },
            { opt.seed,              "-s", "--seed" },
            { to::set(opt.spike_file_output), to::flag, "-f", "--write-spikes" },
            { to::set(opt.profile_only_zero), to::flag, "-z", "--profile-rank-zero" },
            { to::set(opt.verbose),           to::flag, "-v", "--verbose" },
            { to::action(help),               to::flag, to::exit, "-h", "--help" }
    };

    if (!to::run(options, argc, argv+1)) return {};
    if (argv[1]) throw to::option_error("unrecogonized argument", argv[1]);

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
    o << "Simulation options:\n";
    o << "  Excitatory cells                                           : " << options.nexc << "\n";
    o << "  Inhibitory cells                                           : " << options.ninh << "\n";
    o << "  Poisson connections per cell                               : " << options.next << "\n";
    o << "  Proportion of synapses/cell from each population           : " << options.syn_per_cell_prop << "\n";
    o << "  Weight of excitatory synapses                              : " << options.weight << "\n";
    o << "  Relative strength of inhibitory synapses                   : " << options.rel_inh_strength << "\n";
    o << "  Delay of all synapses                                      : " << options.delay << "\n";
    o << "  Expected number of spikes from a single poisson cell per ms: " << options.poiss_lambda << "\n";
    o << "\n";
    o << "  Simulation time                                            : " << options.tfinal << "\n";
    o << "  dt                                                         : " << options.dt << "\n";
    o << "  Group size                                                 : " << options.group_size << "\n";
    o << "  Seed                                                       : " << options.seed << "\n";
    return o;
}
