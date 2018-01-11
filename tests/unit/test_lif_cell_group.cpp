#include "../gtest.h"
#include <cell_group_factory.hpp>
#include <fstream>
#include <lif_cell_description.hpp>
#include <lif_cell_group_mc.hpp>
#include <load_balance.hpp>
#include <model.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>
#include <rss_cell_group.hpp>

using namespace arb;
// Simple ring network of lif neurons.
class ring_recipe: public arb::recipe {
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
        cell_member_type source{(gid + cell_gid_type(ncells_) - 2) % (ncells_ - 1), 0};
        cell_member_type target{gid, 0};
        cell_connection conn(source, target, weight_, delay_);
        connections.push_back(conn);

        // Connect fake cell (numbered ncells_-1) to the first cell (numbered 0).
        if (gid == 0) {
            cell_member_type source{cell_gid_type(ncells_) - 1, 0};
            cell_member_type target{gid, 0};
            cell_connection conn(source, target, weight_, delay_);
            connections.push_back(conn);
        }

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (gid < ncells_ - 1) {
            return lif_cell_description();
        }
        // Produces just a single spike at time 0ms.
        auto rs = arb::rss_cell();
        rs.start_time = 0;
        rs.period = 1;
        rs.stop_time = 0.5;
        return rs;
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
    std::vector<event_generator_ptr> event_generators(cell_gid_type) const override {
        return {};
    }

private:
    int ncells_;
    float weight_, delay_;
};

class path_recipe: public arb::recipe {
public:
    path_recipe(cell_size_type n, float weight, float delay):
    ncells_(n), weight_(weight), delay_(delay)
    {}

    cell_size_type num_cells() const override {
        return ncells_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif_neuron;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid == 0) {
            return {};
        }
        std::vector<cell_connection> connections;
        cell_member_type source{gid - 1, 0};
        cell_member_type target{gid, 0};
        cell_connection conn(source, target, weight_, delay_);
        connections.push_back(conn);

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        return lif_cell_description();
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
    std::vector<event_generator_ptr> event_generators(cell_gid_type) const override {
        return {};
    }

private:
    int ncells_;
    float weight_, delay_;
};

TEST(lif_cell_group_mc, recipe)
{
    ring_recipe rr(100, 1, 0.1);
    EXPECT_EQ(101, rr.num_cells());
    EXPECT_EQ(2, rr.connections_on(0u).size());
    EXPECT_EQ(1, rr.connections_on(55u).size());
    EXPECT_EQ(0, rr.connections_on(1u)[0].source.gid);
    EXPECT_EQ(99, rr.connections_on(0u)[0].source.gid);
}

TEST(lif_cell_group_mc, spikes) {
    // make two lif cells
    path_recipe recipe(2, 1000, 0.1);

    hw::node_info nd;
    nd.num_cpu_cores = threading::num_threads();

    auto decomp = partition_load_balance(recipe, nd);
    model m(recipe, decomp);

    std::vector<postsynaptic_spike_event> events;

    // First event to trigger the spike (first neuron).
    events.push_back({{0, 0}, 1, 1000});

    // This event happens inside the refractory period of the previous
    // event, thus, should be ignored (first neuron)
    events.push_back({{0, 0}, 1.1, 1000});

    // This event happens long after the refractory period of the previous
    // event, should thus trigger new spike (first neuron).
    events.push_back({{0, 0}, 50, 1000});

    m.inject_events(events);

    time_type tfinal = 100;
    time_type dt = 0.01;
    m.run(tfinal, dt);

    // we expect 4 spikes: 2 by both neurons
    EXPECT_EQ(4, m.num_spikes());
}

TEST(lif_cell_group_mc, ring)
{
    // Total number of cells.
    int num_cells = 99;
    double weight = 1000;
    double delay = 1;

    hw::node_info nd;
    nd.num_cpu_cores = threading::num_threads();

    // Total simulation time.
    time_type simulation_time = 100;

    // The number of cells in a single cell group.
    cell_size_type group_size = 10;

    auto recipe = ring_recipe(num_cells, weight, delay);
    auto decomp = partition_load_balance(recipe, nd);

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
    // The total number of cells in all the cell groups.
    EXPECT_EQ((num_cells + 1), recipe.num_cells());

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

