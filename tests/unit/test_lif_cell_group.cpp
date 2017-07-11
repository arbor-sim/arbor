#include "../gtest.h"

#include <cell_group_factory.hpp>
#include <fstream>
#include <lif_cell_description.hpp>
#include <lif_cell_group.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>
#include <rss_cell_group.hpp>

using namespace nest::mc;
// Simple ring network of lif neurons.
class ring_recipe: public nest::mc::recipe {
public:
    ring_recipe(cell_size_type n, float weight, float delay):
    ncells_(n + 1), weight_(weight), delay_(delay)
    {}

    cell_size_type num_cells() const override {
        return ncells_;
    }

    // LIF neurons have gid in range [0..ncells_-2] whereas fake cell is numbered with ncells_ - 1.
    cell_kind get_cell_kind(cell_gid_type gid) const override {
        if (gid < ncells_ - 1) {
            return cell_kind::lif_neuron;
        }
        return cell_kind::regular_spike_source;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid == ncells_ - 1) {
            return {};
        }
        // In a ring, each cell has just one incoming connection.
        std::vector<cell_connection> connections;
        cell_connection conn;
        conn.weight = weight_;
        conn.delay = delay_;
        conn.source = {(gid + cell_gid_type(ncells_) - 2) % (ncells_ - 1), 0};
        conn.dest =   {gid, 0};
        connections.push_back(conn);

        // Connect fake cell (numbered ncells_-1) to the first cell (numbered 0).
        if (gid == 0) {
            conn.weight = weight_;
            conn.delay = delay_;
            conn.source = {cell_gid_type(ncells_) - 1, 0};
            conn.dest =   {gid, 0};
            connections.push_back(conn);
        }

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (gid < ncells_ - 1) {
            return lif_cell_description();
        }
        // Produces just a single spike at time 0ms.
        return rss_cell::rss_cell_description(0, 1, 0.5);
    }

    cell_count_info get_cell_count_info(cell_gid_type) const override {
        return {1u, 1u, 0u};
    }

private:
    int ncells_;
    float weight_, delay_;
};


TEST(lif_cell_group, recipe)
{
    ring_recipe rr(100, 1, 0.1);
    EXPECT_EQ(101, rr.num_cells());
    EXPECT_EQ(2, rr.connections_on(0u).size());
    EXPECT_EQ(1, rr.connections_on(55u).size());
    EXPECT_EQ(0, rr.connections_on(1u)[0].source.gid);
    EXPECT_EQ(99, rr.connections_on(0u)[0].source.gid);
}

TEST(lif_cell_group, cell_group_factory) {
    std::vector<util::unique_any> cells;
    cells.emplace_back(lif_cell_description());
    cells.emplace_back(lif_cell_description());

    cell_group_ptr group = cell_group_factory(
        cell_kind::lif_neuron,
        0,
        cells,
        backend_policy::use_multicore);

    std::vector<postsynaptic_spike_event> events;

    // First event to trigger the spike (first neuron).
    events.push_back({{0, 0}, 1, 1000});

    // This event happens inside the refractory period of the previous
    // event, thus, should be ignored (first neuron)
    events.push_back({{0, 0}, 1.1, 1000});

    // This event happens long after the refractory period of the previous
    // event, should thus trigger new spike (first neuron).
    events.push_back({{0, 0}, 50, 1000});

    // This is event to the second neuron.
    events.push_back({{1, 0}, 1, 1000});

    group->enqueue_events(events);
    group->advance(100, 0.01);
    std::vector<spike> spikes = group->spikes();
    EXPECT_EQ(3, spikes.size());
}

TEST(lif_cell_group, spikes_testing) {
    std::vector<util::unique_any> cells;
    cells.emplace_back(lif_cell_description());

    auto gr = cell_group_factory(cell_kind::lif_neuron, 0, cells, backend_policy::use_multicore);
    auto group = dynamic_cast<lif_cell_group*>(gr.get());

    std::vector<postsynaptic_spike_event> events;
    std::vector<time_type> incoming_spikes;
    time_type simulation_end = 50;
    // add events at times i for the first 80% time of the simulation
    for (size_t i = 1; i < (int) (0.8 * simulation_end); i++) {
        // last parameter is the weight
        events.push_back({{0, 0}, static_cast<time_type>(i), 100});
        incoming_spikes.push_back(i);
    }

    group->enqueue_events(events);
    group->turn_on_sampling(0.01);

    // second parameter is dt, but is ignored
    group->advance(simulation_end, 0.01);
    std::vector<spike> spikes = group->spikes();

    std::vector<std::pair<time_type, double> > voltage = group->voltage();

    std::ofstream in_spikes_file;
    in_spikes_file.open("../../tests/unit/lif_neuron_input_spikes.txt");

    std::ofstream out_spikes_file;
    out_spikes_file.open("../../tests/unit/lif_neuron_output_spikes.txt");

    std::ofstream voltage_file;
    voltage_file.open("../../tests/unit/lif_neuron_voltage.txt");

    for (auto& in_spike : incoming_spikes) {
        in_spikes_file << in_spike << std::endl;
    }

    for (auto& out_spike : spikes) {
        out_spikes_file << out_spike.time << std::endl;
    }

    for (auto& v : voltage) {
        voltage_file << v.first << " " << v.second << std::endl;
    }

    in_spikes_file.close();
    out_spikes_file.close();
    voltage_file.close();
}

TEST(lif_cell_group, domain_decomposition)
{
    // Total number of cells.
    int num_cells = 99;
    double weight = 1000;
    double delay = 1;

    // Total simulation time.
    time_type simulation_time = 100;

    // The number of cells in a single cell group.
    cell_size_type group_size = 10;

    auto recipe = ring_recipe(num_cells, weight, delay);
    // Group rules specifies the number of cells in each cell group
    // and the backend policy.
    group_rules rules{group_size, backend_policy::use_multicore};
    domain_decomposition decomp(recipe, rules);

    // Creates a model with a ring recipe of lif neurons
    model mod(recipe, decomp);

    std::vector<spike> spike_buffer;

    mod.set_global_spike_callback(
        [&spike_buffer](const std::vector<spike>& spikes) {
            spike_buffer.insert(spike_buffer.end(), spikes.begin(), spikes.end());
        }
    );

    // Runs the simulation for simulation_time with given timestep
    mod.run(simulation_time, 0.01);
    // The number of cell groups.
    EXPECT_EQ(11, mod.num_groups());
    // The total number of cells in all the cell groups.
    EXPECT_EQ((num_cells + 1), mod.num_cells());

    // Since delay is 1, we expect to see spike in each second.
    //EXPECT_EQ(simulation_time + 1, mod.num_spikes());

    for (auto& spike : spike_buffer) {
        // Assumes that delay = 1
        // We expect that Regular Spiking Cell spiked at time 0s.
        if (spike.source.gid == num_cells) {
            EXPECT_EQ(0, spike.time);
        // Other LIF cell should spike consecutively.
        } else {
            EXPECT_EQ(spike.source.gid + 1, spike.time);
        }
    }
}

