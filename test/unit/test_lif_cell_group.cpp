#include <gtest/gtest.h>

#include "common.hpp"

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_source_cell.hpp>

#include "lif_cell_group.hpp"

using namespace arb;
// Simple ring network of LIF neurons.
// with one regularly spiking cell (fake cell) connected to the first cell in the ring.
class ring_recipe: public arb::recipe {
public:
    ring_recipe(cell_size_type n_lif_cells, float weight, float delay):
        n_lif_cells_(n_lif_cells), weight_(weight), delay_(delay)
    {}

    cell_size_type num_cells() const override {
        return n_lif_cells_ + 1;
    }

    // LIF neurons have gid in range [1..n_lif_cells_] whereas fake cell is numbered with 0.
    cell_kind get_cell_kind(cell_gid_type gid) const override {
        if (gid == 0) {
            return cell_kind::spike_source;
        }
        return cell_kind::lif;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid == 0) {
            return {};
        }

        // In a ring, each cell has just one incoming connection.
        std::vector<cell_connection> connections;
        // gid-1 >= 0 since gid != 0
        auto src_gid = (gid - 1) % n_lif_cells_;
        cell_connection conn({src_gid, "src"}, {"tgt"}, weight_, delay_);
        connections.push_back(conn);

        // If first LIF cell, then add
        // the connection from the last LIF cell as well
        if (gid == 1) {
            auto src_gid = n_lif_cells_;
            cell_connection conn({src_gid, "src"}, {"tgt"}, weight_, delay_);
            connections.push_back(conn);
        }

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        // regularly spiking cell.
        if (gid == 0) {
            // Produces just a single spike at time 0ms.
            return spike_source_cell("src", explicit_schedule({0.f}));
        }
        // LIF cell.
        auto cell = lif_cell("src", "tgt");
        return cell;
    }

private:
    cell_size_type n_lif_cells_;
    float weight_, delay_;
};

// LIF cells connected in the manner of a path 0->1->...->n-1.
class path_recipe: public arb::recipe {
public:
    path_recipe(cell_size_type n, float weight, float delay):
        ncells_(n), weight_(weight), delay_(delay)
    {}

    cell_size_type num_cells() const override {
        return ncells_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        if (gid == 0) {
            return {};
        }
        std::vector<cell_connection> connections;
        cell_connection conn({gid-1, "src"}, {"tgt"}, weight_, delay_);
        connections.push_back(conn);

        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        auto cell = lif_cell("src", "tgt");
        return cell;
    }

private:
    cell_size_type ncells_;
    float weight_, delay_;
};

// LIF cell with probe
class probe_recipe: public arb::recipe {
public:
    probe_recipe() {}

    cell_size_type num_cells() const override {
        return 2;
    }
    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif;
    }
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        return {};
    }
    util::unique_any get_cell_description(cell_gid_type gid) const override {
        auto cell = lif_cell("src", "tgt");
        if (gid == 0) {
            cell.E_L = -42;
            cell.V_m = -23;
            cell.t_ref = 0.2;
        }
        return cell;
    }
    std::vector<probe_info> get_probes(cell_gid_type gid) const override {
        if (gid == 0) {
            return {arb::lif_probe_voltage{}, arb::lif_probe_voltage{}};
        } else {
            return {arb::lif_probe_voltage{}};
        }
    }
    std::vector<event_generator> event_generators(cell_gid_type) const override { return {regular_generator({"tgt"}, 100.0, 0.25, 0.05)}; }
};

TEST(lif_cell_group, throw) {
    probe_recipe rec;
    auto context = make_context();
    auto decomp = partition_load_balance(rec, context);
    EXPECT_NO_THROW(simulation(rec, context, decomp));
}

TEST(lif_cell_group, recipe)
{
    ring_recipe rr(100, 1, 0.1);
    EXPECT_EQ(101u, rr.num_cells());
    EXPECT_EQ(2u, rr.connections_on(1u).size());
    EXPECT_EQ(1u, rr.connections_on(55u).size());
    EXPECT_EQ(0u, rr.connections_on(1u)[0].source.gid);
    EXPECT_EQ(100u, rr.connections_on(1u)[1].source.gid);
}

TEST(lif_cell_group, spikes) {
    // make two lif cells
    path_recipe recipe(2, 1000, 0.1);

    auto context = make_context();

    auto decomp = partition_load_balance(recipe, context);
    simulation sim(recipe, context, decomp);

    cse_vector events;

    // First event to trigger the spike (first neuron).
    events.push_back({0, {{0, 1, 1000}}});

    // This event happens inside the refractory period of the previous
    // event, thus, should be ignored (first neuron)
    events.push_back({0, {{0, 1.1, 1000}}});

    // This event happens long after the refractory period of the previous
    // event, should thus trigger new spike (first neuron).
    events.push_back({0, {{0, 50, 1000}}});

    sim.inject_events(events);

    time_type tfinal = 100;
    time_type dt = 0.01;
    sim.run(tfinal, dt);

    // we expect 4 spikes: 2 by both neurons
    EXPECT_EQ(4u, sim.num_spikes());
}

TEST(lif_cell_group, ring)
{
    // Total number of LIF cells.
    cell_size_type num_lif_cells = 99;
    double weight = 1000;
    double delay = 1;

    // Total simulation time.
    time_type simulation_time = 100;

    auto recipe = ring_recipe(num_lif_cells, weight, delay);
    // Creates a simulation with a ring recipe of lif neurons
    simulation sim(recipe);

    std::vector<spike> spike_buffer;

    sim.set_global_spike_callback(
        [&spike_buffer](const std::vector<spike>& spikes) {
            spike_buffer.insert(spike_buffer.end(), spikes.begin(), spikes.end());
        }
    );

    // Runs the simulation for simulation_time with given timestep
    sim.run(simulation_time, 0.01);
    // The total number of cells in all the cell groups.
    // There is one additional fake cell (regularly spiking cell).
    EXPECT_EQ(num_lif_cells + 1u, recipe.num_cells());

    for (auto& spike : spike_buffer) {
        // Assumes that delay = 1
        // We expect that Regular Spiking Cell spiked at time 0s.
        if (spike.source.gid == 0) {
            EXPECT_EQ(0, spike.time);
        // Other LIF cell should spike consecutively.
        } else {
            EXPECT_EQ(spike.source.gid, spike.time);
        }
    }
}

struct Um_type {
    constexpr static double delta = 1e-6;

    double t;
    double u;

    friend std::ostream& operator<<(std::ostream& os, const Um_type& um) {
        os << "{ " << um.t << ", " << um.u << " }";
        return os;
    }

    friend bool operator==(const Um_type& lhs, const Um_type& rhs) {
        return (std::abs(lhs.t - rhs.t) <= delta)
            && (std::abs(lhs.u - rhs.u) <= delta);
    }
};

TEST(lif_cell_group, probe) {
    auto ums = std::unordered_map<cell_member_type, std::vector<Um_type>>{};
    auto fun = [&ums](probe_metadata pm,
                  std::size_t n,
                  const sample_record* samples) {
        for (int ix = 0; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *util::any_cast<double*>(v);
            ums[pm.id].push_back({t, u});
        }
    };
    auto rec = probe_recipe{};
    auto sim = simulation(rec);

    sim.add_sampler(all_probes, regular_schedule(0.025), fun);
    sim.run(1.5, 0.005);
    std::vector<Um_type> exp = {{0, -23},
        {0.025, -22.9425718},
        {0.05, -22.885287},
        {0.075, -22.8281453},
        {0.1, -22.7711462},
        {0.125, -22.7142894},
        {0.15, -22.6575746},
        {0.175, -22.6010014},
        {0.2, -22.5445695},
        {0.225, -22.4882785},
        {0.25, -17.432128},
        {0.275, -17.3886021},
        {0.3, -12.3451849},
        {0.325, -12.3143605},
        {0.35, -7.28361301},
        {0.375, -7.26542672},
        {0.4, -2.24728584},
        {0.425, -2.24167464},
        {0.45, 2.76392255},
        {0.475, 2.75702137},
        {0.5, 7.75013743},
        {0.525, 7.73078628},
        {0.55, -42},
        {0.575, -42},
        {0.6, -42},
        {0.625, -42},
        {0.65, -42},
        {0.675, -42},
        {0.7, -42},
        {0.725, -42},
        {0.75, -37},
        {0.775, -36.9076155},
        {0.8, -31.8154617},
        {0.825, -31.7360224},
        {0.85, -26.6567815},
        {0.875, -26.5902227},
        {0.9, -21.5238302},
        {0.925, -21.4700878},
        {0.95, -16.4164796},
        {0.975, -16.3754897},
        {1, -11.3346021},
        {1.025, -11.306301},
        {1.05, -6.27807055},
        {1.075, -6.26239498},
        {1.1, -1.24675854},
        {1.125, -1.24364554},
        {1.15, 3.75945969},
        {1.175, 3.75007278},
        {1.2, 8.74070931},
        {1.225, 8.71888483},
        {1.25, -42},
        {1.275, -42},
        {1.3, -42},
        {1.325, -42},
        {1.35, -42},
        {1.375, -42},
        {1.4, -42},
        {1.425, -42},
        {1.45, -37},
        {1.475, -36.9076155},};

    ASSERT_TRUE(testing::seq_eq(ums[{0, 0}], exp));
    ASSERT_TRUE(testing::seq_eq(ums[{0, 1}], exp));
    // gid == 1 is different
    ASSERT_FALSE(testing::seq_eq(ums[{1, 0}], exp));
}
