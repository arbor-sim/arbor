#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/distributed_context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/threadinfo.hpp>
#include <arbor/version.hpp>

#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#include <aux/path.hpp>
#include <aux/spike_emitter.hpp>
#include <aux/strsub.hpp>
#ifdef ARB_MPI_ENABLED
#include <aux/with_mpi.hpp>
#endif

#include "io.hpp"

using namespace arb;

void banner(domain_info, const distributed_context*);

// Samples m unique values in interval [start, end) - gid.
// We exclude gid because we don't want self-loops.
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
        ncells_exc_(nexc), ncells_inh_(ninh), ncells_ext_(next), delay_(delay), seed_(seed) {
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
        return cell_kind::lif_neuron;
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

        using pgen = poisson_generator<std::mt19937_64>;

        time_type t0 = 0;
        cell_member_type target{gid, 0};

        gens.emplace_back(pgen(target, weight_ext_, G, t0, lambda_));
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

    // Number of Poisson connections each neuron receives 
    cell_size_type ncells_ext_;

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
    distributed_context context;

    try {
#ifdef ARB_MPI_ENABLED
        with_mpi guard(argc, argv, false);
        context = mpi_context(MPI_COMM_WORLD);
#endif
        arb::profile::meter_manager meters(&context);
        meters.start();
        std::cout << aux::mask_stream(context.id()==0);
        // read parameters
        io::cl_options options = io::read_options(argc, argv, context.id()==0);
        domain_info nd = local_domain_info();
        banner(nd, &context);

        meters.checkpoint("setup");

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
        hints[cell_kind::lif_neuron].cpu_group_size = group_size;
        auto decomp = partition_load_balance(recipe, nd, &context, hints);

        simulation sim(recipe, decomp, &context);

        // Initialize the spike exporting interface
        std::fstream spike_out;
        if (options.spike_file_output) {
            using std::ios_base;

            auto rank = context.id();
            aux::path p = options.output_path;
            p /= aux::strsub("%_%.%", options.file_name, rank, options.file_extension);

            if (options.single_file_per_rank) {
                spike_out = aux::open_or_throw(p, ios_base::out, !options.over_write);
                sim.set_local_spike_callback(aux::spike_emitter(spike_out));
            }
            else if (rank==0) {
                spike_out = aux::open_or_throw(p, ios_base::out, !options.over_write);
                sim.set_global_spike_callback(aux::spike_emitter(spike_out));
            }
        }

        meters.checkpoint("model-init");

        // run simulation
        sim.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate");

        // output profile and diagnostic feedback
        std::cout << profile::profiler_summary() << "\n";
        std::cout << "\nThere were " << sim.num_spikes() << " spikes\n";

        auto report = profile::make_meter_report(meters);
        std::cout << report;
        if (context.id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << aux::to_json(report) << "\n";
        }
    }
    catch (io::usage_error& e) {
        // only print usage/startup errors on master
        std::cerr << aux::mask_stream(context.id()==0);
        std::cerr << e.what() << "\n";
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    return 0;
}

void banner(domain_info nd, const distributed_context* ctx) {
    std::cout << "==========================================\n";
    std::cout << "  Arbor miniapp\n";
    std::cout << "  - distributed : " << ctx->size()
              << " (" << ctx->name() << ")\n";
    std::cout << "  - threads     : " << nd.num_threads
              << " (" << arb::thread_implementation() << ")\n";
    std::cout << "  - gpus        : " << nd.num_gpus << "\n";
    std::cout << "==========================================\n";
}

