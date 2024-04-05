#include <gtest/gtest.h>

#include "common.hpp"

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/adex_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/units.hpp>

namespace U = arb::units;
using namespace U::literals;

namespace {
// Simple ring network of ADEX neurons.
// with one regularly spiking cell (fake cell) connected to the first cell in the ring.
struct ring_recipe: public arb::recipe {
    ring_recipe(arb::cell_size_type n_adex_cells, float weight, const U::quantity& delay):
        n_adex_cells_(n_adex_cells), weight_(weight), delay_(delay)
    {}

    arb::cell_size_type num_cells() const override { return n_adex_cells_ + 1; }

    // ADEX neurons have gid in range [1..n_adex_cells_] whereas fake cell is numbered with 0.
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override {
        if (gid == 0) return arb::cell_kind::spike_source;
        return arb::cell_kind::adex;
    }

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        if (gid == 0) return {};

        // In a ring, each cell has just one incoming connection.
        // gid-1 >= 0 since gid != 0
        auto src_gid = (gid - 1) % n_adex_cells_;
        std::vector<arb::cell_connection> connections;
        connections.emplace_back(arb::cell_global_label_type{src_gid, "src"}, arb::cell_local_label_type{"tgt"}, weight_, delay_);

        // If first ADEX cell, then connect from the last ADEX cell, too
        if (gid == 1) {
            auto src_gid = n_adex_cells_;
            connections.emplace_back(arb::cell_global_label_type{src_gid, "src"}, arb::cell_local_label_type{"tgt"}, weight_, delay_);
        }
        return connections;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        // regularly spiking cell; produces a single spike at time 0.
        if (gid == 0) return arb::spike_source_cell("src",
                                                    arb::explicit_schedule({0.0_ms}));
        return arb::adex_cell{"src", "tgt"};
    }

private:
    arb::cell_size_type n_adex_cells_;
    float weight_;
    U::quantity delay_ = 42.0_ms;
};

// ADEX cells connected in the manner of a path 0->1->...->n-1.
struct path_recipe: public arb::recipe {
    path_recipe(arb::cell_size_type n, float weight, const U::quantity& delay):
        ncells_(n), weight_(weight), delay_(delay) {}

    arb::cell_size_type num_cells() const override { return ncells_; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return arb::cell_kind::adex; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override { return arb::adex_cell("src", "tgt"); }

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        if (gid == 0) return {};
        return {{{gid-1, "src"}, {"tgt"}, weight_, delay_}};
    }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        if (gid != 0) return {};
        return {arb::explicit_generator_from_milliseconds({"tgt"},
                                                          1000.0,           // use a large weight to trigger spikes
                                                          std::vector{ 1.0, // First event to trigger the spike
                                                                       1.1, // inside refractory period; should be ignored
                                                                       50.0 // long after previous event; should trigger new spike
                                                  })};
    }


private:
    arb::cell_size_type ncells_;
    float weight_;
    U::quantity delay_;
};

// ADEX cell with probe
struct adex_probe_recipe: public arb::recipe {
    adex_probe_recipe(std::size_t n_conn = 0): n_conn_{n_conn} {}

    arb::cell_size_type num_cells() const override { return 2; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return arb::cell_kind::adex; }
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        return {n_conn_,
                {arb::cell_global_label_type{0, "src"},
                 arb::cell_local_label_type{"tgt"},
                 0.0,
                 0.005_ms}};
    }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        auto cell = arb::adex_cell("src", "tgt");
        cell.V_th = 10_mV;
        if (0 == gid) {
            cell.E_R = -1*23.0_mV;
            cell.V_m = -1*18.0_mV;
            cell.E_L = -1*13.0_mV;
            cell.t_ref = 0.8_ms;
            cell.tau = 5*U::ms;
        }
        return cell;
    }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override {
        return {{arb::adex_probe_voltage{}, "a"}};
    }
    std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return {
            arb::regular_generator({"tgt"},
                                   200.0,
                                   2.0_ms,
                                   1.0_ms,
                                   6.0_ms)
        };
    }

    std::size_t n_conn_ = 0;
};
} // namespace

TEST(adex_cell_group, throw) {
    adex_probe_recipe rec;
    auto context = arb::make_context();
    auto decomp = partition_load_balance(rec, context);
    EXPECT_NO_THROW(arb::simulation(rec, context, decomp));
}


TEST(adex_cell_group, recipe)
{
    ring_recipe rr(100, 1, 0.1_ms);
    EXPECT_EQ(101u, rr.num_cells());
    EXPECT_EQ(0u, rr.connections_on(0u).size());
    EXPECT_EQ(1u, rr.connections_on(55u).size());
    auto conns_gid_1 = rr.connections_on(1);
    EXPECT_EQ(2u,   conns_gid_1.size());
    EXPECT_EQ(0u,   conns_gid_1[0].source.gid);
    EXPECT_EQ(100u, conns_gid_1[1].source.gid);
}

TEST(adex_cell_group, spikes) {
    // make two adex cells
    path_recipe recipe(2, 1000, 0.1_ms);
    arb::simulation sim(recipe);
    sim.run(100_ms, 0.01_ms);
    // we expect 4 spikes: 2 by both neurons
    EXPECT_EQ(4u, sim.num_spikes());
}

TEST(adex_cell_group, ring)
{
    // Total number of ADEX cells.
    arb::cell_size_type num_adex_cells = 99;
    double weight = 1000;
    auto delay = 1.0_ms;
    auto recipe = ring_recipe(num_adex_cells, weight, delay);
    // Creates a simulation with a ring recipe of adex neurons
    arb::simulation sim(recipe);

    std::vector<arb::spike> spike_buffer;

    sim.set_global_spike_callback(
        [&spike_buffer](const std::vector<arb::spike>& spikes) {
            spike_buffer.insert(spike_buffer.end(), spikes.begin(), spikes.end());
        }
    );

    // Runs the simulation for simulation_time with given timestep
    sim.run(100_ms, 0.01_ms);
    // The total number of cells in all the cell groups.
    // There is one additional fake cell (regularly spiking cell).
    EXPECT_EQ(num_adex_cells + 1u, recipe.num_cells());

    for (auto& spike: spike_buffer) {
        // Assumes that delay = 1
        // We expect that Regular Spiking Cell spiked at time 0s.
        if (spike.source.gid == 0) {
            EXPECT_EQ(0, spike.time);
        // Other ADEX cell should spike consecutively.
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

TEST(adex_cell_group, probe) {
    auto ums = std::unordered_map<arb::cell_address_type, std::vector<Um_type>>{};
    auto fun = [&ums](arb::probe_metadata pm,
                      std::size_t n,
                      const arb::sample_record* samples) {
        for (std::size_t ix = 0; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            EXPECT_NE(arb::util::any_cast<const double*>(v), nullptr);
            double u = *arb::util::any_cast<const double*>(v);
            ums[pm.id].push_back({t, u});
        }
    };
    auto rec = adex_probe_recipe{};
    auto sim = arb::simulation(rec);

    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.025_ms), fun);

    std::vector<arb::spike> spikes;

    sim.set_global_spike_callback([&spikes](const std::vector<arb::spike>& spk) {
        for (const auto& s: spk) spikes.push_back(s);
    });

    sim.run(10.0_ms, 0.005_ms);
    std::vector<Um_type> exp = {{ 0, -18 },
                                { 0.025, -17.9866178 },
                                { 0.05, -17.9732626 },
                                { 0.075, -17.9599343 },
                                { 0.1, -17.946633 },
                                { 0.125, -17.9333585 },
                                { 0.15, -17.920111 },
                                { 0.175, -17.9068904 },
                                { 0.2, -17.8936968 },
                                { 0.225, -17.88053 },
                                { 0.25, -17.8673901 },
                                { 0.275, -17.8542771 },
                                { 0.3, -17.841191 },
                                { 0.325, -17.8281318 },
                                { 0.35, -17.8150995 },
                                { 0.375, -17.802094 },
                                { 0.4, -17.7891153 },
                                { 0.425, -17.7761635 },
                                { 0.45, -17.7632386 },
                                { 0.475, -17.7503404 },
                                { 0.5, -17.7374691 },
                                { 0.525, -17.7246246 },
                                { 0.55, -17.7118069 },
                                { 0.575, -17.6990159 },
                                { 0.6, -17.6862517 },
                                { 0.625, -17.6735143 },
                                { 0.65, -17.6608036 },
                                { 0.675, -17.6481197 },
                                { 0.7, -17.6354625 },
                                { 0.725, -17.622832 },
                                { 0.75, -17.6102282 },
                                { 0.775, -17.597651 },
                                { 0.8, -17.5851005 },
                                { 0.825, -17.5725767 },
                                { 0.85, -17.5600796 },
                                { 0.875, -17.547609 },
                                { 0.9, -17.5351651 },
                                { 0.925, -17.5227477 },
                                { 0.95, -17.5103569 },
                                { 0.975, -17.4979927 },
                                { 1, -17.4856551 },
                                { 1.025, -17.4733439 },
                                { 1.05, -17.4610593 },
                                { 1.075, -17.4488012 },
                                { 1.1, -17.4365696 },
                                { 1.125, -17.4243644 },
                                { 1.15, -17.4121856 },
                                { 1.175, -17.4000333 },
                                { 1.2, -17.3879074 },
                                { 1.225, -17.3758079 },
                                { 1.25, -17.3637347 },
                                { 1.275, -17.3516879 },
                                { 1.3, -17.3396675 },
                                { 1.325, -17.3276733 },
                                { 1.35, -17.3157055 },
                                { 1.375, -17.3037639 },
                                { 1.4, -17.2918485 },
                                { 1.425, -17.2799594 },
                                { 1.45, -17.2680965 },
                                { 1.475, -17.2562598 },
                                { 1.5, -17.2444492 },
                                { 1.525, -17.2326649 },
                                { 1.55, -17.2209066 },
                                { 1.575, -17.2091744 },
                                { 1.6, -17.1974683 },
                                { 1.625, -17.1857883 },
                                { 1.65, -17.1741343 },
                                { 1.675, -17.1625063 },
                                { 1.7, -17.1509043 },
                                { 1.725, -17.1393283 },
                                { 1.75, -17.1277782 },
                                { 1.775, -17.116254 },
                                { 1.8, -17.1047557 },
                                { 1.825, -17.0932833 },
                                { 1.85, -17.0818368 },
                                { 1.875, -17.070416 },
                                { 1.9, -17.0590211 },
                                { 1.925, -17.0476519 },
                                { 1.95, -17.0363085 },
                                { 1.975, -17.0249908 },
                                { 2, -17.0136988 },
                                { 2.025, -23 },
                                { 2.05, -23 },
                                { 2.075, -23 },
                                { 2.1, -23 },
                                { 2.125, -23 },
                                { 2.15, -23 },
                                { 2.175, -23 },
                                { 2.2, -23 },
                                { 2.225, -23 },
                                { 2.25, -23 },
                                { 2.275, -23 },
                                { 2.3, -23 },
                                { 2.325, -23 },
                                { 2.35, -23 },
                                { 2.375, -23 },
                                { 2.4, -23 },
                                { 2.425, -23 },
                                { 2.45, -23 },
                                { 2.475, -23 },
                                { 2.5, -23 },
                                { 2.525, -23 },
                                { 2.55, -23 },
                                { 2.575, -23 },
                                { 2.6, -23 },
                                { 2.625, -23 },
                                { 2.65, -23 },
                                { 2.675, -23 },
                                { 2.7, -23 },
                                { 2.725, -23 },
                                { 2.75, -23 },
                                { 2.775, -23 },
                                { 2.8, -23 },
                                { 2.825, -22.9798329 },
                                { 2.85, -22.9596691 },
                                { 2.875, -22.9395089 },
                                { 2.9, -22.9193525 },
                                { 2.925, -22.8992003 },
                                { 2.95, -22.8790525 },
                                { 2.975, -22.8589094 },
                                { 3, -22.8387712 },
                                { 3.025, -23 },
                                { 3.05, -23 },
                                { 3.075, -23 },
                                { 3.1, -23 },
                                { 3.125, -23 },
                                { 3.15, -23 },
                                { 3.175, -23 },
                                { 3.2, -23 },
                                { 3.225, -23 },
                                { 3.25, -23 },
                                { 3.275, -23 },
                                { 3.3, -23 },
                                { 3.325, -23 },
                                { 3.35, -23 },
                                { 3.375, -23 },
                                { 3.4, -23 },
                                { 3.425, -23 },
                                { 3.45, -23 },
                                { 3.475, -23 },
                                { 3.5, -23 },
                                { 3.525, -23 },
                                { 3.55, -23 },
                                { 3.575, -23 },
                                { 3.6, -23 },
                                { 3.625, -23 },
                                { 3.65, -23 },
                                { 3.675, -23 },
                                { 3.7, -23 },
                                { 3.725, -23 },
                                { 3.75, -23 },
                                { 3.775, -23 },
                                { 3.8, -23 },
                                { 3.825, -22.9865565 },
                                { 3.85, -22.9730647 },
                                { 3.875, -22.9595252 },
                                { 3.9, -22.9459387 },
                                { 3.925, -22.9323056 },
                                { 3.95, -22.9186265 },
                                { 3.975, -22.904902 },
                                { 4, -22.8911327 },
                                { 4.025, -23 },
                                { 4.05, -23 },
                                { 4.075, -23 },
                                { 4.1, -23 },
                                { 4.125, -23 },
                                { 4.15, -23 },
                                { 4.175, -23 },
                                { 4.2, -23 },
                                { 4.225, -23 },
                                { 4.25, -23 },
                                { 4.275, -23 },
                                { 4.3, -23 },
                                { 4.325, -23 },
                                { 4.35, -23 },
                                { 4.375, -23 },
                                { 4.4, -23 },
                                { 4.425, -23 },
                                { 4.45, -23 },
                                { 4.475, -23 },
                                { 4.5, -23 },
                                { 4.525, -23 },
                                { 4.55, -23 },
                                { 4.575, -23 },
                                { 4.6, -23 },
                                { 4.625, -23 },
                                { 4.65, -23 },
                                { 4.675, -23 },
                                { 4.7, -23 },
                                { 4.725, -23 },
                                { 4.75, -23 },
                                { 4.775, -23 },
                                { 4.8, -23 },
                                { 4.825, -22.9930159 },
                                { 4.85, -22.9859341 },
                                { 4.875, -22.9787553 },
                                { 4.9, -22.9714804 },
                                { 4.925, -22.9641104 },
                                { 4.95, -22.9566459 },
                                { 4.975, -22.9490879 },
                                { 5, -22.9414372 },
                                { 5.025, -23 },
                                { 5.05, -23 },
                                { 5.075, -23 },
                                { 5.1, -23 },
                                { 5.125, -23 },
                                { 5.15, -23 },
                                { 5.175, -23 },
                                { 5.2, -23 },
                                { 5.225, -23 },
                                { 5.25, -23 },
                                { 5.275, -23 },
                                { 5.3, -23 },
                                { 5.325, -23 },
                                { 5.35, -23 },
                                { 5.375, -23 },
                                { 5.4, -23 },
                                { 5.425, -23 },
                                { 5.45, -23 },
                                { 5.475, -23 },
                                { 5.5, -23 },
                                { 5.525, -23 },
                                { 5.55, -23 },
                                { 5.575, -23 },
                                { 5.6, -23 },
                                { 5.625, -23 },
                                { 5.65, -23 },
                                { 5.675, -23 },
                                { 5.7, -23 },
                                { 5.725, -23 },
                                { 5.75, -23 },
                                { 5.775, -23 },
                                { 5.8, -23 },
                                { 5.825, -22.9992216 },
                                { 5.85, -22.9982979 },
                                { 5.875, -22.9972299 },
                                { 5.9, -22.9960188 },
                                { 5.925, -22.9946657 },
                                { 5.95, -22.9931718 },
                                { 5.975, -22.991538 },
                                { 6, -22.9897656 },
                                { 6.025, -22.9878556 },
                                { 6.05, -22.985809 },
                                { 6.075, -22.983627 },
                                { 6.1, -22.9813106 },
                                { 6.125, -22.9788609 },
                                { 6.15, -22.976279 },
                                { 6.175, -22.9735658 },
                                { 6.2, -22.9707225 },
                                { 6.225, -22.9677501 },
                                { 6.25, -22.9646496 },
                                { 6.275, -22.961422 },
                                { 6.3, -22.9580684 },
                                { 6.325, -22.9545897 },
                                { 6.35, -22.9509871 },
                                { 6.375, -22.9472614 },
                                { 6.4, -22.9434137 },
                                { 6.425, -22.939445 },
                                { 6.45, -22.9353563 },
                                { 6.475, -22.9311485 },
                                { 6.5, -22.9268226 },
                                { 6.525, -22.9223796 },
                                { 6.55, -22.9178205 },
                                { 6.575, -22.9131462 },
                                { 6.6, -22.9083576 },
                                { 6.625, -22.9034558 },
                                { 6.65, -22.8984416 },
                                { 6.675, -22.893316 },
                                { 6.7, -22.8880799 },
                                { 6.725, -22.8827342 },
                                { 6.75, -22.8772799 },
                                { 6.775, -22.8717179 },
                                { 6.8, -22.866049 },
                                { 6.825, -22.8602743 },
                                { 6.85, -22.8543945 },
                                { 6.875, -22.8484106 },
                                { 6.9, -22.8423234 },
                                { 6.925, -22.8361339 },
                                { 6.95, -22.8298429 },
                                { 6.975, -22.8234514 },
                                { 7, -22.81696 },
                                { 7.025, -22.8103698 },
                                { 7.05, -22.8036816 },
                                { 7.075, -22.7968962 },
                                { 7.1, -22.7900145 },
                                { 7.125, -22.7830373 },
                                { 7.15, -22.7759654 },
                                { 7.175, -22.7687998 },
                                { 7.2, -22.7615411 },
                                { 7.225, -22.7541903 },
                                { 7.25, -22.7467481 },
                                { 7.275, -22.7392154 },
                                { 7.3, -22.7315929 },
                                { 7.325, -22.7238815 },
                                { 7.35, -22.716082 },
                                { 7.375, -22.7081951 },
                                { 7.4, -22.7002217 },
                                { 7.425, -22.6921625 },
                                { 7.45, -22.6840183 },
                                { 7.475, -22.6757899 },
                                { 7.5, -22.6674781 },
                                { 7.525, -22.6590836 },
                                { 7.55, -22.6506071 },
                                { 7.575, -22.6420494 },
                                { 7.6, -22.6334114 },
                                { 7.625, -22.6246937 },
                                { 7.65, -22.615897 },
                                { 7.675, -22.6070221 },
                                { 7.7, -22.5980698 },
                                { 7.725, -22.5890407 },
                                { 7.75, -22.5799357 },
                                { 7.775, -22.5707553 },
                                { 7.8, -22.5615004 },
                                { 7.825, -22.5521716 },
                                { 7.85, -22.5427696 },
                                { 7.875, -22.5332952 },
                                { 7.9, -22.523749 },
                                { 7.925, -22.5141317 },
                                { 7.95, -22.5044441 },
                                { 7.975, -22.4946868 },
                                { 8, -22.4848605 },
                                { 8.025, -22.4749658 },
                                { 8.05, -22.4650035 },
                                { 8.075, -22.4549742 },
                                { 8.1, -22.4448785 },
                                { 8.125, -22.4347172 },
                                { 8.15, -22.4244909 },
                                { 8.175, -22.4142002 },
                                { 8.2, -22.4038458 },
                                { 8.225, -22.3934283 },
                                { 8.25, -22.3829484 },
                                { 8.275, -22.3724067 },
                                { 8.3, -22.3618039 },
                                { 8.325, -22.3511405 },
                                { 8.35, -22.3404173 },
                                { 8.375, -22.3296347 },
                                { 8.4, -22.3187935 },
                                { 8.425, -22.3078942 },
                                { 8.45, -22.2969374 },
                                { 8.475, -22.2859239 },
                                { 8.5, -22.2748541 },
                                { 8.525, -22.2637286 },
                                { 8.55, -22.2525481 },
                                { 8.575, -22.2413131 },
                                { 8.6, -22.2300243 },
                                { 8.625, -22.2186822 },
                                { 8.65, -22.2072874 },
                                { 8.675, -22.1958404 },
                                { 8.7, -22.1843419 },
                                { 8.725, -22.1727924 },
                                { 8.75, -22.1611925 },
                                { 8.775, -22.1495427 },
                                { 8.8, -22.1378436 },
                                { 8.825, -22.1260957 },
                                { 8.85, -22.1142997 },
                                { 8.875, -22.102456 },
                                { 8.9, -22.0905651 },
                                { 8.925, -22.0786277 },
                                { 8.95, -22.0666443 },
                                { 8.975, -22.0546154 },
                                { 9, -22.0425415 },
                                { 9.025, -22.0304232 },
                                { 9.05, -22.0182609 },
                                { 9.075, -22.0060553 },
                                { 9.1, -21.9938067 },
                                { 9.125, -21.9815158 },
                                { 9.15, -21.969183 },
                                { 9.175, -21.9568089 },
                                { 9.2, -21.944394 },
                                { 9.225, -21.9319387 },
                                { 9.25, -21.9194435 },
                                { 9.275, -21.906909 },
                                { 9.3, -21.8943356 },
                                { 9.325, -21.8817238 },
                                { 9.35, -21.8690742 },
                                { 9.375, -21.8563871 },
                                { 9.4, -21.8436631 },
                                { 9.425, -21.8309026 },
                                { 9.45, -21.8181062 },
                                { 9.475, -21.8052742 },
                                { 9.5, -21.7924072 },
                                { 9.525, -21.7795056 },
                                { 9.55, -21.7665699 },
                                { 9.575, -21.7536005 },
                                { 9.6, -21.7405979 },
                                { 9.625, -21.7275625 },
                                { 9.65, -21.7144948 },
                                { 9.675, -21.7013952 },
                                { 9.7, -21.6882642 },
                                { 9.725, -21.6751022 },
                                { 9.75, -21.6619096 },
                                { 9.775, -21.6486869 },
                                { 9.8, -21.6354345 },
                                { 9.825, -21.6221529 },
                                { 9.85, -21.6088424 },
                                { 9.875, -21.5955035 },
                                { 9.9, -21.5821366 },
                                { 9.925, -21.5687421 },
                                { 9.95, -21.5553205 },
                                { 9.975, -21.5418721 },};

    ASSERT_TRUE(testing::seq_eq(ums[{0, "a"}], exp));
    // gid == 1 is different, but of same size
    EXPECT_EQ((ums[{1, "a"}].size()), exp.size());
    ASSERT_FALSE(testing::seq_eq(ums[{1, "a"}], exp));
    // now check the spikes
    std::sort(spikes.begin(), spikes.end());
    EXPECT_EQ(spikes.size(), 6u);
    std::vector<arb::spike> sexp{{{0, 0}, 2}, {{0, 0}, 3}, {{0, 0}, 4}, {{0, 0}, 5},
                                 {{1, 0}, 2}, {{1, 0}, 5}, };
    ASSERT_EQ(spikes, sexp);
}

TEST(adex_cell_group, probe_with_connections) {
    auto ums = std::unordered_map<arb::cell_address_type, std::vector<Um_type>>{};
    auto fun = [&ums](arb::probe_metadata pm,
                      std::size_t n,
                      const arb::sample_record* samples) {
        for (std::size_t ix = 0; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *arb::util::any_cast<const double*>(v);
            ums[pm.id].push_back({t, u});
        }
    };
    auto rec = adex_probe_recipe{5};
    auto sim = arb::simulation(rec);

    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.025_ms), fun);

    std::vector<arb::spike> spikes;

    sim.set_global_spike_callback(
        [&spikes](const std::vector<arb::spike>& spk) { for (const auto& s: spk) spikes.push_back(s); }
    );

    sim.run(10.0_ms, 0.005_ms);

    std::vector<Um_type> exp = {{ 0, -18 },
                                { 0.025, -17.9866192 },
                                { 0.05, -17.9732653 },
                                { 0.075, -17.9599383 },
                                { 0.1, -17.9466383 },
                                { 0.125, -17.9333653 },
                                { 0.15, -17.9201191 },
                                { 0.175, -17.9068998 },
                                { 0.2, -17.8937075 },
                                { 0.225, -17.8805421 },
                                { 0.25, -17.8674036 },
                                { 0.275, -17.8542919 },
                                { 0.3, -17.8412072 },
                                { 0.325, -17.8281493 },
                                { 0.35, -17.8151183 },
                                { 0.375, -17.8021141 },
                                { 0.4, -17.7891368 },
                                { 0.425, -17.7761864 },
                                { 0.45, -17.7632627 },
                                { 0.475, -17.7503659 },
                                { 0.5, -17.7374959 },
                                { 0.525, -17.7246527 },
                                { 0.55, -17.7118363 },
                                { 0.575, -17.6990467 },
                                { 0.6, -17.6862839 },
                                { 0.625, -17.6735478 },
                                { 0.65, -17.6608384 },
                                { 0.675, -17.6481558 },
                                { 0.7, -17.6354999 },
                                { 0.725, -17.6228707 },
                                { 0.75, -17.6102682 },
                                { 0.775, -17.5976924 },
                                { 0.8, -17.5851432 },
                                { 0.825, -17.5726207 },
                                { 0.85, -17.5601248 },
                                { 0.875, -17.5476556 },
                                { 0.9, -17.5352129 },
                                { 0.925, -17.5227969 },
                                { 0.95, -17.5104074 },
                                { 0.975, -17.4980445 },
                                { 1, -17.4857081 },
                                { 1.025, -17.4733983 },
                                { 1.05, -17.461115 },
                                { 1.075, -17.4488581 },
                                { 1.1, -17.4366278 },
                                { 1.125, -17.4244239 },
                                { 1.15, -17.4122464 },
                                { 1.175, -17.4000954 },
                                { 1.2, -17.3879707 },
                                { 1.225, -17.3758725 },
                                { 1.25, -17.3638006 },
                                { 1.275, -17.3517551 },
                                { 1.3, -17.3397359 },
                                { 1.325, -17.327743 },
                                { 1.35, -17.3157764 },
                                { 1.375, -17.3038361 },
                                { 1.4, -17.291922 },
                                { 1.425, -17.2800341 },
                                { 1.45, -17.2681725 },
                                { 1.475, -17.256337 },
                                { 1.5, -17.2445277 },
                                { 1.525, -17.2327445 },
                                { 1.55, -17.2209875 },
                                { 1.575, -17.2092566 },
                                { 1.6, -17.1975517 },
                                { 1.625, -17.1858729 },
                                { 1.65, -17.1742201 },
                                { 1.675, -17.1625934 },
                                { 1.7, -17.1509926 },
                                { 1.725, -17.1394178 },
                                { 1.75, -17.1278689 },
                                { 1.775, -17.1163459 },
                                { 1.8, -17.1048489 },
                                { 1.825, -17.0933777 },
                                { 1.85, -17.0819323 },
                                { 1.875, -17.0705128 },
                                { 1.9, -17.059119 },
                                { 1.925, -17.047751 },
                                { 1.95, -17.0364088 },
                                { 1.975, -17.0250923 },
                                { 2, -17.013855 },
                                { 2.025, -23 },
                                { 2.05, -23 },
                                { 2.075, -23 },
                                { 2.1, -23 },
                                { 2.125, -23 },
                                { 2.15, -23 },
                                { 2.175, -23 },
                                { 2.2, -23 },
                                { 2.225, -23 },
                                { 2.25, -23 },
                                { 2.275, -23 },
                                { 2.3, -23 },
                                { 2.325, -23 },
                                { 2.35, -23 },
                                { 2.375, -23 },
                                { 2.4, -23 },
                                { 2.425, -23 },
                                { 2.45, -23 },
                                { 2.475, -23 },
                                { 2.5, -23 },
                                { 2.525, -23 },
                                { 2.55, -23 },
                                { 2.575, -23 },
                                { 2.6, -23 },
                                { 2.625, -23 },
                                { 2.65, -23 },
                                { 2.675, -23 },
                                { 2.7, -23 },
                                { 2.725, -23 },
                                { 2.75, -23 },
                                { 2.775, -23 },
                                { 2.8, -23 },
                                { 2.825, -22.9798333 },
                                { 2.85, -22.9596698 },
                                { 2.875, -22.93951 },
                                { 2.9, -22.9193541 },
                                { 2.925, -22.8992023 },
                                { 2.95, -22.8790549 },
                                { 2.975, -22.8589123 },
                                { 3, -22.8387768 },
                                { 3.025, -23 },
                                { 3.05, -23 },
                                { 3.075, -23 },
                                { 3.1, -23 },
                                { 3.125, -23 },
                                { 3.15, -23 },
                                { 3.175, -23 },
                                { 3.2, -23 },
                                { 3.225, -23 },
                                { 3.25, -23 },
                                { 3.275, -23 },
                                { 3.3, -23 },
                                { 3.325, -23 },
                                { 3.35, -23 },
                                { 3.375, -23 },
                                { 3.4, -23 },
                                { 3.425, -23 },
                                { 3.45, -23 },
                                { 3.475, -23 },
                                { 3.5, -23 },
                                { 3.525, -23 },
                                { 3.55, -23 },
                                { 3.575, -23 },
                                { 3.6, -23 },
                                { 3.625, -23 },
                                { 3.65, -23 },
                                { 3.675, -23 },
                                { 3.7, -23 },
                                { 3.725, -23 },
                                { 3.75, -23 },
                                { 3.775, -23 },
                                { 3.8, -23 },
                                { 3.825, -22.9865544 },
                                { 3.85, -22.9730605 },
                                { 3.875, -22.959519 },
                                { 3.9, -22.9459305 },
                                { 3.925, -22.9322955 },
                                { 3.95, -22.9186145 },
                                { 3.975, -22.9048882 },
                                { 4, -22.8911191 },
                                { 4.025, -23 },
                                { 4.05, -23 },
                                { 4.075, -23 },
                                { 4.1, -23 },
                                { 4.125, -23 },
                                { 4.15, -23 },
                                { 4.175, -23 },
                                { 4.2, -23 },
                                { 4.225, -23 },
                                { 4.25, -23 },
                                { 4.275, -23 },
                                { 4.3, -23 },
                                { 4.325, -23 },
                                { 4.35, -23 },
                                { 4.375, -23 },
                                { 4.4, -23 },
                                { 4.425, -23 },
                                { 4.45, -23 },
                                { 4.475, -23 },
                                { 4.5, -23 },
                                { 4.525, -23 },
                                { 4.55, -23 },
                                { 4.575, -23 },
                                { 4.6, -23 },
                                { 4.625, -23 },
                                { 4.65, -23 },
                                { 4.675, -23 },
                                { 4.7, -23 },
                                { 4.725, -23 },
                                { 4.75, -23 },
                                { 4.775, -23 },
                                { 4.8, -23 },
                                { 4.825, -22.9930115 },
                                { 4.85, -22.9859252 },
                                { 4.875, -22.9787422 },
                                { 4.9, -22.9714631 },
                                { 4.925, -22.9640889 },
                                { 4.95, -22.9566204 },
                                { 4.975, -22.9490585 },
                                { 5, -22.9414052 },
                                { 5.025, -23 },
                                { 5.05, -23 },
                                { 5.075, -23 },
                                { 5.1, -23 },
                                { 5.125, -23 },
                                { 5.15, -23 },
                                { 5.175, -23 },
                                { 5.2, -23 },
                                { 5.225, -23 },
                                { 5.25, -23 },
                                { 5.275, -23 },
                                { 5.3, -23 },
                                { 5.325, -23 },
                                { 5.35, -23 },
                                { 5.375, -23 },
                                { 5.4, -23 },
                                { 5.425, -23 },
                                { 5.45, -23 },
                                { 5.475, -23 },
                                { 5.5, -23 },
                                { 5.525, -23 },
                                { 5.55, -23 },
                                { 5.575, -23 },
                                { 5.6, -23 },
                                { 5.625, -23 },
                                { 5.65, -23 },
                                { 5.675, -23 },
                                { 5.7, -23 },
                                { 5.725, -23 },
                                { 5.75, -23 },
                                { 5.775, -23 },
                                { 5.8, -23 },
                                { 5.825, -22.999215 },
                                { 5.85, -22.9982847 },
                                { 5.875, -22.9972104 },
                                { 5.9, -22.995993 },
                                { 5.925, -22.9946337 },
                                { 5.95, -22.9931337 },
                                { 5.975, -22.9914939 },
                                { 6, -22.9897156 },
                                { 6.025, -22.9877998 },
                                { 6.05, -22.9857476 },
                                { 6.075, -22.98356 },
                                { 6.1, -22.9812381 },
                                { 6.125, -22.978783 },
                                { 6.15, -22.9761958 },
                                { 6.175, -22.9734774 },
                                { 6.2, -22.970629 },
                                { 6.225, -22.9676516 },
                                { 6.25, -22.9645462 },
                                { 6.275, -22.9613138 },
                                { 6.3, -22.9579554 },
                                { 6.325, -22.9544722 },
                                { 6.35, -22.950865 },
                                { 6.375, -22.9471348 },
                                { 6.4, -22.9432828 },
                                { 6.425, -22.9393098 },
                                { 6.45, -22.9352169 },
                                { 6.475, -22.931005 },
                                { 6.5, -22.9266751 },
                                { 6.525, -22.9222281 },
                                { 6.55, -22.9176652 },
                                { 6.575, -22.9129871 },
                                { 6.6, -22.9081948 },
                                { 6.625, -22.9032894 },
                                { 6.65, -22.8982716 },
                                { 6.675, -22.8931426 },
                                { 6.7, -22.8879031 },
                                { 6.725, -22.8825541 },
                                { 6.75, -22.8770966 },
                                { 6.775, -22.8715315 },
                                { 6.8, -22.8658595 },
                                { 6.825, -22.8600818 },
                                { 6.85, -22.8541991 },
                                { 6.875, -22.8482124 },
                                { 6.9, -22.8421225 },
                                { 6.925, -22.8359303 },
                                { 6.95, -22.8296367 },
                                { 6.975, -22.8232425 },
                                { 7, -22.8167487 },
                                { 7.025, -22.8101561 },
                                { 7.05, -22.8034656 },
                                { 7.075, -22.7966779 },
                                { 7.1, -22.789794 },
                                { 7.125, -22.7828146 },
                                { 7.15, -22.7757407 },
                                { 7.175, -22.768573 },
                                { 7.2, -22.7613125 },
                                { 7.225, -22.7539598 },
                                { 7.25, -22.7465158 },
                                { 7.275, -22.7389813 },
                                { 7.3, -22.7313572 },
                                { 7.325, -22.7236442 },
                                { 7.35, -22.7158431 },
                                { 7.375, -22.7079547 },
                                { 7.4, -22.6999799 },
                                { 7.425, -22.6919193 },
                                { 7.45, -22.6837738 },
                                { 7.475, -22.6755441 },
                                { 7.5, -22.6672311 },
                                { 7.525, -22.6588354 },
                                { 7.55, -22.6503578 },
                                { 7.575, -22.6417992 },
                                { 7.6, -22.6331601 },
                                { 7.625, -22.6244415 },
                                { 7.65, -22.6156439 },
                                { 7.675, -22.6067683 },
                                { 7.7, -22.5978152 },
                                { 7.725, -22.5887854 },
                                { 7.75, -22.5796797 },
                                { 7.775, -22.5704987 },
                                { 7.8, -22.5612432 },
                                { 7.825, -22.5519139 },
                                { 7.85, -22.5425115 },
                                { 7.875, -22.5330367 },
                                { 7.9, -22.5234901 },
                                { 7.925, -22.5138726 },
                                { 7.95, -22.5041847 },
                                { 7.975, -22.4944272 },
                                { 8, -22.4846007 },
                                { 8.025, -22.4747059 },
                                { 8.05, -22.4647435 },
                                { 8.075, -22.4547141 },
                                { 8.1, -22.4446185 },
                                { 8.125, -22.4344573 },
                                { 8.15, -22.424231 },
                                { 8.175, -22.4139405 },
                                { 8.2, -22.4035863 },
                                { 8.225, -22.3931691 },
                                { 8.25, -22.3826894 },
                                { 8.275, -22.3721481 },
                                { 8.3, -22.3615456 },
                                { 8.325, -22.3508826 },
                                { 8.35, -22.3401598 },
                                { 8.375, -22.3293777 },
                                { 8.4, -22.318537 },
                                { 8.425, -22.3076382 },
                                { 8.45, -22.2966821 },
                                { 8.475, -22.2856692 },
                                { 8.5, -22.2746 },
                                { 8.525, -22.2634753 },
                                { 8.55, -22.2522955 },
                                { 8.575, -22.2410613 },
                                { 8.6, -22.2297733 },
                                { 8.625, -22.2184321 },
                                { 8.65, -22.2070381 },
                                { 8.675, -22.1955921 },
                                { 8.7, -22.1840945 },
                                { 8.725, -22.172546 },
                                { 8.75, -22.1609471 },
                                { 8.775, -22.1492983 },
                                { 8.8, -22.1376003 },
                                { 8.825, -22.1258536 },
                                { 8.85, -22.1140587 },
                                { 8.875, -22.1022161 },
                                { 8.9, -22.0903265 },
                                { 8.925, -22.0783904 },
                                { 8.95, -22.0664082 },
                                { 8.975, -22.0543806 },
                                { 9, -22.042308 },
                                { 9.025, -22.030191 },
                                { 9.05, -22.0180302 },
                                { 9.075, -22.0058259 },
                                { 9.1, -21.9935788 },
                                { 9.125, -21.9812894 },
                                { 9.15, -21.9689581 },
                                { 9.175, -21.9565855 },
                                { 9.2, -21.9441721 },
                                { 9.225, -21.9317183 },
                                { 9.25, -21.9192248 },
                                { 9.275, -21.9066919 },
                                { 9.3, -21.8941201 },
                                { 9.325, -21.88151 },
                                { 9.35, -21.8688621 },
                                { 9.375, -21.8561767 },
                                { 9.4, -21.8434545 },
                                { 9.425, -21.8306958 },
                                { 9.45, -21.8179011 },
                                { 9.475, -21.805071 },
                                { 9.5, -21.7922058 },
                                { 9.525, -21.779306 },
                                { 9.55, -21.7663722 },
                                { 9.575, -21.7534047 },
                                { 9.6, -21.740404 },
                                { 9.625, -21.7273705 },
                                { 9.65, -21.7143048 },
                                { 9.675, -21.7012071 },
                                { 9.7, -21.6880781 },
                                { 9.725, -21.6749181 },
                                { 9.75, -21.6617276 },
                                { 9.775, -21.648507 },
                                { 9.8, -21.6352567 },
                                { 9.825, -21.6219771 },
                                { 9.85, -21.6086687 },
                                { 9.875, -21.595332 },
                                { 9.9, -21.5819672 },
                                { 9.925, -21.5685749 },
                                { 9.95, -21.5551554 },
                                { 9.975, -21.5417092 },};

    EXPECT_EQ((ums[{0, "a"}].size()), exp.size());
    ASSERT_TRUE(testing::seq_eq(ums[{0, "a"}], exp));
    // gid == 1 is different, but of same size
    EXPECT_EQ((ums[{1, "a"}].size()), exp.size());
    ASSERT_FALSE(testing::seq_eq(ums[{1, "a"}], exp));
    // now check the spikes
    std::sort(spikes.begin(), spikes.end());
    EXPECT_EQ(spikes.size(), 6u);
    std::vector<arb::spike> sexp{{{0, 0}, 2}, {{0, 0}, 3}, {{0, 0}, 4}, {{0, 0}, 5},
                                 {{1, 0}, 2}, {{1, 0}, 5}, };
    ASSERT_EQ(spikes, sexp);
}
