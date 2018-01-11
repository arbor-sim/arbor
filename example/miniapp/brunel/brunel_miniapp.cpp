#include <cmath>
#include <exception>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <json/json.hpp>
#include <cell_group_factory.hpp>
#include <common_types.hpp>
#include <communication/communicator.hpp>
#include <communication/global_policy.hpp>
#include <cell.hpp>
#include <io/exporter_spike_file.hpp>
#include <lif_cell_description.hpp>
#include <load_balance.hpp>
#include <model.hpp>
#include <profiling/profiler.hpp>
#include <profiling/meter_manager.hpp>
#include <recipe.hpp>
#include <set>
#include <threading/threading.hpp>
#include <util/config.hpp>
#include <util/debug.hpp>
#include <util/ioutil.hpp>
#include <util/nop.hpp>
#include <vector>
#include "io.hpp"
#include <hardware/gpu.hpp>
#include <hardware/node_info.hpp>

using namespace arb;

using global_policy = communication::global_policy;
using file_export_type = io::exporter_spike_file<global_policy>;
void banner(hw::node_info);
using communicator_type = communication::communicator<communication::global_policy>;

// Samples m unique values in interval [start, end) - gid.
// We exclude gid because we don't want self-loops.
std::vector<int> sample_subset(int gid, int start, int end, int m) {
    std::set<int> s;
    std::mt19937 gen(gid + 42);
    std::uniform_int_distribution<int> dis(start, end - 1);
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
   Poisson neurons all spike independently with mean frequency poiss_rate [kHz].
   Because of the refractory period, the activity is mostly driven by Poisson neurons and
   recurrent connections have a small effect.
 */
class brunel_recipe: public recipe {
public:
    brunel_recipe(cell_size_type nexc, cell_size_type ninh, cell_size_type next, double in_degree_prop,
                  float weight, float delay, float rel_inh_strength, double poiss_rate):
        ncells_exc_(nexc), ncells_inh_(ninh), ncells_ext_(next), delay_(delay), rate_(poiss_rate) {
        // Make sure that in_degree_prop in the interval (0, 1]
        if (in_degree_prop <= 0.0 || in_degree_prop > 1.0) {
            std::out_of_range("The proportion of incoming connections should be in the interval (0, 1].");
        }

        // Set up the parameters.
        weight_exc_ = weight;
        weight_inh_ = -rel_inh_strength * weight_exc_;
        weight_ext_ = weight;
        in_degree_exc_ = std::round(in_degree_prop * nexc);
        in_degree_inh_ = std::round(in_degree_prop * ninh);
    }

    cell_size_type num_cells() const override {
        return ncells_exc_ + ncells_inh_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif_neuron;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        // Sample and add incoming excitatory connections.
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
        auto cell = lif_cell_description();
        cell.tau_m = 10;
        cell.V_th = 10;
        cell.C_m = 20;
        cell.E_L = 0;
        cell.V_m = 0;
        cell.V_reset = 0;
        cell.t_ref = 2;
        cell.n_poiss = ncells_ext_;
        cell.w_poiss = weight_ext_;
        cell.d_poiss = delay_;
        cell.rate = rate_;
        return cell;
    }

    std::vector<event_generator_ptr> event_generators(cell_gid_type) const override {
        return {};
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

    // Mean rate of Poisson spiking neurons.
    double rate_;
};

using util::any_cast;
using util::make_span;
using global_policy = communication::global_policy;
using file_export_type = io::exporter_spike_file<global_policy>;

int main(int argc, char** argv) {
    arb::communication::global_policy_guard global_guard(argc, argv);
    try {
        arb::util::meter_manager meters;
        meters.start();
        std::cout << util::mask_stream(global_policy::id()==0);
        // read parameters
        io::cl_options options = io::read_options(argc, argv, global_policy::id()==0);
        hw::node_info nd;
        nd.num_cpu_cores = threading::num_threads();
        nd.num_gpus = hw::num_gpus()>0? 1: 0;
        banner(nd);

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

        // Mean rate of Poisson cells [kHz].
        double poiss_rate = options.poiss_rate;

        // The number of cells in a single cell group.
        cell_size_type group_size = options.group_size;

        brunel_recipe recipe(nexc, ninh, next, in_degree_prop, w, d, rel_inh_strength, poiss_rate);

        auto register_exporter = [] (const io::cl_options& options) {
            return util::make_unique<file_export_type>
                       (options.file_name, options.output_path,
                        options.file_extension, options.over_write);
        };

        auto decomp = partition_load_balance(recipe, nd);
        model m(recipe, decomp);

        // Initialize the spike exporting interface
        std::unique_ptr<file_export_type> file_exporter;
        if (options.spike_file_output) {
            if (options.single_file_per_rank) {
                file_exporter = register_exporter(options);

                m.set_local_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                        file_exporter->output(spikes);
                    }
                );
            }
            else if(communication::global_policy::id()==0) {
                file_exporter = register_exporter(options);

                m.set_global_spike_callback(
                    [&](const std::vector<spike>& spikes) {
                        file_exporter->output(spikes);
                    }
                );
            }
        }
        meters.checkpoint("model-init");

        // run model
        m.run(options.tfinal, options.dt);

        meters.checkpoint("model-simulate");

        // output profile and diagnostic feedback
        auto const num_steps = options.tfinal / options.dt;
        util::profiler_output(0.001, options.profile_only_zero);
        std::cout << "there were " << m.num_spikes() << " spikes\n";

        auto report = util::make_meter_report(meters);
        std::cout << report;
        if (global_policy::id()==0) {
            std::ofstream fid;
            fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
            fid.open("meters.json");
            fid << std::setw(1) << util::to_json(report) << "\n";
        }
    }
    catch (io::usage_error& e) {
        // only print usage/startup errors on master
        std::cerr << util::mask_stream(global_policy::id()==0);
        std::cerr << e.what() << "\n";
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << "\n";
        return 2;
    }
    return 0;
}

void banner(hw::node_info nd) {
    std::cout << "==========================================\n";
    std::cout << "  Arbor miniapp\n";
    std::cout << "  - distributed : " << global_policy::size()
              << " (" << std::to_string(global_policy::kind()) << ")\n";
    std::cout << "  - threads     : " << nd.num_cpu_cores
              << " (" << threading::description() << ")\n";
    std::cout << "  - gpus        : " << nd.num_gpus << "\n";
    std::cout << "==========================================\n";
}

