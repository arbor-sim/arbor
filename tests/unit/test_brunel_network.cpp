#include "../gtest.h"

#include <cell_group_factory.hpp>
#include <fstream>
#include <lif_cell_description.hpp>
#include <lif_cell_group.hpp>
#include <model.hpp>
#include <pss_cell_description.hpp>
#include <pss_cell_group.hpp>
#include <recipe.hpp>
#include <util/span.hpp>

using namespace nest::mc;

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
     Brunel networks consists of nexc excitatory LIF neurons, ninh inhibitory LIF neurons and
     nexc Poisson neurons. Each neuron in the network receives in_degree_prop * nexc excitatory connections
     chosen randomly, in_degree_prop * ninh inhibitory connections and in_degree_prop * nexc Poisson connections.
     All the connections have the same delay. The strenght of excitatory and poisson connections is given by 
     parameter weight, whereas the strength of inhibitory connections is rel_inh_strength * weight. 
     Poisson neurons all spike independently with mean frequency poiss_rate [kHz].
     Because of the refractory period, the activity is mostly driven by poisson neurons and 
     recurrent connections have just a small effect.
 */
class brunel_recipe: public recipe {
public:
    brunel_recipe(cell_size_type nexc, cell_size_type ninh, double in_degree_prop, float weight, float delay, float rel_inh_strength, double poiss_rate):
    ncells_exc_(nexc), ncells_inh_(ninh), delay_(delay), rate_(poiss_rate)
    {
        // Make sure that in_degree_prop in the interval (0, 1]
        if (in_degree_prop <= 0.0 || in_degree_prop > 1.0) {
            std::out_of_range("The proportion of incoming connections should be in the interval (0, 1].");
        }

        // Set up the parameters.
        ncells_ext_ = nexc;
        weight_exc_ = weight;
        weight_inh_ = -1.0 * rel_inh_strength * weight_exc_;
        weight_ext_ = weight;
        in_degree_exc_ = (int) (in_degree_prop * nexc);
        in_degree_inh_ = (int) (in_degree_prop * ninh);
        in_degree_ext_ = (int) (in_degree_prop * ncells_ext_);
    }

    cell_size_type num_cells() const override {
        return ncells_exc_ + ncells_inh_ + ncells_ext_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        if (gid < ncells_exc_ + ncells_inh_) {
            return cell_kind::lif_neuron;
        }

        return cell_kind::poisson_spike_source;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        // Sample and add incoming excitatory connections.
        for (auto i: sample_subset(gid, 0, ncells_exc_, in_degree_exc_)) {
            cell_connection conn;
            conn.source = {cell_gid_type(i), 0};
            conn.dest = {gid, 0};
            conn.weight = weight_exc_;
            conn.delay = delay_;

            connections.push_back(conn);
        }

        // Add incoming inhibitory connections.
        for (auto i: sample_subset(gid, ncells_exc_, ncells_exc_ + ncells_inh_, in_degree_inh_)) {
            cell_connection conn;
            conn.source = {cell_gid_type(i), 0};
            conn.dest = {gid, 0};
            conn.weight = weight_inh_;
            conn.delay = delay_;

            connections.push_back(conn);
        }

        // Add incoming external Poisson connections.
        for (auto i: sample_subset(gid, ncells_exc_ + ncells_inh_, ncells_exc_ + ncells_inh_ + ncells_ext_, in_degree_ext_)) {
            cell_connection conn;
            conn.source = {cell_gid_type(i), 0};
            conn.dest = {gid, 0};
            conn.weight = weight_ext_;
            conn.delay = delay_;

            connections.push_back(conn);
        }

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (gid < ncells_exc_ + ncells_inh_) {
            auto cell = lif_cell_description();
            cell.tau_m = 10;
            cell.V_th = 10;
            cell.C_m = 20;
            cell.E_L = 0;
            cell.V_m = 0;
            cell.V_reset = 0;
            cell.t_ref = 2;
            return cell;
        }
        return pss_cell_description(rate_);
    }

    cell_count_info get_cell_count_info(cell_gid_type) const override {
        return {1u, 1u, 0u};
    }

private:
    // Number of excitatory cells.
    cell_size_type ncells_exc_;

    // Number of inhibitory cells.
    cell_size_type ncells_inh_;

    // Number of cells in the external Poisson population
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

    // Number of connections that each neuron receives from the Poisson population.
    int in_degree_ext_;

    // Mean rate of Poisson spiking neurons.
    double rate_;
};

TEST(pss_cell_group, brunels_network) {
    // The size of excitatory and poisson population.
    cell_size_type nexc = 400;

    // The size of inhibitory population.
    cell_size_type ninh = 100;

    // Fraction of connections each neuron receives from each of the 3 populations.
    double in_degree_prop = 0.05;

    // Weight of excitatory and poisson connections.
    float w = 1.2;

    // Delay of all the connections.
    float d = 0.1;

    // Relative strength of inhibitory connections with respect to excitatory connections.
    float rel_inh_strength = 1;

    // Mean rate of Poisson cells [kHz].
    double poiss_rate = 1;

    // The number of cells in a single cell group.
    cell_size_type group_size = 10;

    brunel_recipe recipe(nexc, ninh, in_degree_prop, w, d, rel_inh_strength, poiss_rate);

    // Group rules specifies the number of cells in each cell group
    // and the backend policy.
    group_rules rules{group_size, backend_policy::use_multicore};
    domain_decomposition decomp(recipe, rules);

    // Creates a model with a brunel's recipe.
    model mod(recipe, decomp);
    std::vector<spike> spike_buffer;

    // Adds a callback function that collects all the spikes into a vector.
    mod.set_global_spike_callback(
        [&spike_buffer](const std::vector<spike>& spikes) {
            spike_buffer.insert(spike_buffer.end(), spikes.begin(), spikes.end());
        }
    );

    // Runs the simulation.
    mod.run(50, 1);

    std::vector<std::vector<time_type> > spike_times(nexc);

    // Distribute the spikes of excitatory population
    // according to its gid.
    for (auto& spike : spike_buffer) {
        auto source = spike.source.gid;
        auto time = spike.time;

        if (source < nexc) {
            spike_times[source].push_back(time);
        }
    }

    for (auto& vec : spike_times) {
        //std::cout << vec.size() << std::endl;
        time_type avg_inter_spike_time = 0;
        for (auto i = 1; i < vec.size(); ++i) {
            // Check if the time difference between two consecutive spikes of each LIF neuron
            // is never less than 2ms, since that is refractory period.
            EXPECT_TRUE(vec[i] - vec[i-1] >= 2);
            avg_inter_spike_time += vec[i] - vec[i - 1];
        }
        // The average inter spike time should be uniformly distributed
        // in roughly this interval.
        avg_inter_spike_time /= vec.size() - 1;
        EXPECT_TRUE(avg_inter_spike_time >= 10.0);
        EXPECT_TRUE(avg_inter_spike_time <= 25.0);
    }
};

