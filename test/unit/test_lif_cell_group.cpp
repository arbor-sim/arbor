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
            cell.E_R = -23;
            cell.V_m = -18;
            cell.E_L = -13;
            cell.t_ref = 0.8;
            cell.tau_m = 5;
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
    std::vector<event_generator> event_generators(cell_gid_type) const override {
        return {regular_generator({"tgt"}, 200.0, 2.0, 1.0, 6.0)};
    }
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
        for (std::size_t ix = 0; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *util::any_cast<double*>(v);
            ums[pm.id].push_back({t, u});
        }
    };
    auto rec = probe_recipe{};
    auto sim = simulation(rec);

    sim.add_sampler(all_probes, regular_schedule(0.025), fun);

    std::vector<double> spikes;

    sim.set_global_spike_callback(
        [&spikes](const std::vector<spike>& spk) { for (const auto& s: spk) spikes.push_back(s.time); }
    );

    sim.run(10, 0.005);
std::vector<Um_type> exp = {{ 0, -18 },
                            { 0.025, -17.9750624 },
                            { 0.05, -17.9502492 },
                            { 0.075, -17.9255597 },
                            { 0.1, -17.9009934 },
                            { 0.125, -17.8765496 },
                            { 0.15, -17.8522277 },
                            { 0.175, -17.8280271 },
                            { 0.2, -17.8039472 },
                            { 0.225, -17.7799874 },
                            { 0.25, -17.7561471 },
                            { 0.275, -17.7324257 },
                            { 0.3, -17.7088227 },
                            { 0.325, -17.6853373 },
                            { 0.35, -17.6619691 },
                            { 0.375, -17.6387174 },
                            { 0.4, -17.6155817 },
                            { 0.425, -17.5925614 },
                            { 0.45, -17.5696559 },
                            { 0.475, -17.5468647 },
                            { 0.5, -17.5241871 },
                            { 0.525, -17.5016226 },
                            { 0.55, -17.4791707 },
                            { 0.575, -17.4568307 },
                            { 0.6, -17.4346022 },
                            { 0.625, -17.4124845 },
                            { 0.65, -17.3904772 },
                            { 0.675, -17.3685796 },
                            { 0.7, -17.3467912 },
                            { 0.725, -17.3251115 },
                            { 0.75, -17.3035399 },
                            { 0.775, -17.2820759 },
                            { 0.8, -17.2607189 },
                            { 0.825, -17.2394685 },
                            { 0.85, -17.2183241 },
                            { 0.875, -17.1972851 },
                            { 0.9, -17.1763511 },
                            { 0.925, -17.1555214 },
                            { 0.95, -17.1347957 },
                            { 0.975, -17.1141733 },
                            { 1, -17.0936538 },
                            { 1.025, -17.0732366 },
                            { 1.05, -17.0529212 },
                            { 1.075, -17.0327072 },
                            { 1.1, -17.012594 },
                            { 1.125, -16.9925811 },
                            { 1.15, -16.972668 },
                            { 1.175, -16.9528542 },
                            { 1.2, -16.9331393 },
                            { 1.225, -16.9135227 },
                            { 1.25, -16.8940039 },
                            { 1.275, -16.8745825 },
                            { 1.3, -16.8552579 },
                            { 1.325, -16.8360297 },
                            { 1.35, -16.8168975 },
                            { 1.375, -16.7978606 },
                            { 1.4, -16.7789187 },
                            { 1.425, -16.7600713 },
                            { 1.45, -16.7413178 },
                            { 1.475, -16.7226579 },
                            { 1.5, -16.7040911 },
                            { 1.525, -16.6856169 },
                            { 1.55, -16.6672348 },
                            { 1.575, -16.6489444 },
                            { 1.6, -16.6307452 },
                            { 1.625, -16.6126368 },
                            { 1.65, -16.5946187 },
                            { 1.675, -16.5766904 },
                            { 1.7, -16.5588516 },
                            { 1.725, -16.5411018 },
                            { 1.75, -16.5234404 },
                            { 1.775, -16.5058672 },
                            { 1.8, -16.4883816 },
                            { 1.825, -16.4709833 },
                            { 1.85, -16.4536717 },
                            { 1.875, -16.4364464 },
                            { 1.9, -16.419307 },
                            { 1.925, -16.4022532 },
                            { 1.95, -16.3852844 },
                            { 1.975, -16.3684002 },
                            { 2, -6.35160023 },
                            { 2.025, -6.38475926 },
                            { 2.05, -6.41775291 },
                            { 2.075, -6.45058201 },
                            { 2.1, -6.48324737 },
                            { 2.125, -6.51574981 },
                            { 2.15, -6.54809014 },
                            { 2.175, -6.58026917 },
                            { 2.2, -6.61228771 },
                            { 2.225, -6.64414656 },
                            { 2.25, -6.67584651 },
                            { 2.275, -6.70738836 },
                            { 2.3, -6.73877289 },
                            { 2.325, -6.77000089 },
                            { 2.35, -6.80107314 },
                            { 2.375, -6.83199042 },
                            { 2.4, -6.8627535 },
                            { 2.425, -6.89336314 },
                            { 2.45, -6.92382012 },
                            { 2.475, -6.95412519 },
                            { 2.5, -6.98427912 },
                            { 2.525, -7.01428265 },
                            { 2.55, -7.04413654 },
                            { 2.575, -7.07384153 },
                            { 2.6, -7.10339837 },
                            { 2.625, -7.1328078 },
                            { 2.65, -7.16207054 },
                            { 2.675, -7.19118733 },
                            { 2.7, -7.22015891 },
                            { 2.725, -7.24898599 },
                            { 2.75, -7.27766929 },
                            { 2.775, -7.30620953 },
                            { 2.8, -7.33460743 },
                            { 2.825, -7.36286369 },
                            { 2.85, -7.39097903 },
                            { 2.875, -7.41895414 },
                            { 2.9, -7.44678972 },
                            { 2.925, -7.47448647 },
                            { 2.95, -7.50204508 },
                            { 2.975, -7.52946625 },
                            { 3, 2.44324935 },
                            { 3.025, 2.36622582 },
                            { 3.05, 2.28958645 },
                            { 3.075, 2.21332932 },
                            { 3.1, 2.13745252 },
                            { 3.125, 2.06195417 },
                            { 3.15, 1.98683236 },
                            { 3.175, 1.91208522 },
                            { 3.2, 1.83771088 },
                            { 3.225, 1.76370749 },
                            { 3.25, 1.69007319 },
                            { 3.275, 1.61680615 },
                            { 3.3, 1.54390452 },
                            { 3.325, 1.47136649 },
                            { 3.35, 1.39919025 },
                            { 3.375, 1.32737399 },
                            { 3.4, 1.25591592 },
                            { 3.425, 1.18481424 },
                            { 3.45, 1.11406718 },
                            { 3.475, 1.04367298 },
                            { 3.5, 0.973629868 },
                            { 3.525, 0.903936098 },
                            { 3.55, 0.834589928 },
                            { 3.575, 0.765589623 },
                            { 3.6, 0.696933458 },
                            { 3.625, 0.628619717 },
                            { 3.65, 0.560646693 },
                            { 3.675, 0.493012686 },
                            { 3.7, 0.425716004 },
                            { 3.725, 0.358754966 },
                            { 3.75, 0.292127898 },
                            { 3.775, 0.225833133 },
                            { 3.8, 0.159869015 },
                            { 3.825, 0.0942338948 },
                            { 3.85, 0.0289261308 },
                            { 3.875, -0.0360559094 },
                            { 3.9, -0.10071385 },
                            { 3.925, -0.165049308 },
                            { 3.95, -0.229063892 },
                            { 3.975, -0.292759202 },
                            { 4, 9.64386317 },
                            { 4.025, 9.53092643 },
                            { 4.05, 9.41855297 },
                            { 4.075, 9.30673997 },
                            { 4.1, 9.19548464 },
                            { 4.125, 9.0847842 },
                            { 4.15, 8.97463588 },
                            { 4.175, 8.86503692 },
                            { 4.2, 8.7559846 },
                            { 4.225, 8.64747617 },
                            { 4.25, 8.53950893 },
                            { 4.275, 8.43208018 },
                            { 4.3, 8.32518724 },
                            { 4.325, 8.21882742 },
                            { 4.35, 8.11299808 },
                            { 4.375, 8.00769656 },
                            { 4.4, 7.90292024 },
                            { 4.425, 7.79866649 },
                            { 4.45, 7.69493271 },
                            { 4.475, 7.5917163 },
                            { 4.5, 7.48901469 },
                            { 4.525, 7.3868253 },
                            { 4.55, 7.28514558 },
                            { 4.575, 7.183973 },
                            { 4.6, 7.08330501 },
                            { 4.625, 6.98313911 },
                            { 4.65, 6.88347279 },
                            { 4.675, 6.78430355 },
                            { 4.7, 6.68562893 },
                            { 4.725, 6.58744644 },
                            { 4.75, 6.48975365 },
                            { 4.775, 6.3925481 },
                            { 4.8, 6.29582736 },
                            { 4.825, 6.19958902 },
                            { 4.85, 6.10383067 },
                            { 4.875, 6.00854992 },
                            { 4.9, 5.91374438 },
                            { 4.925, 5.81941168 },
                            { 4.95, 5.72554948 },
                            { 4.975, 5.63215541 },
                            { 5, -23 },
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
                            { 5.825, -22.9501248 },
                            { 5.85, -22.9004983 },
                            { 5.875, -22.8511194 },
                            { 5.9, -22.8019867 },
                            { 5.925, -22.7530991 },
                            { 5.95, -22.7044553 },
                            { 5.975, -22.6560542 },
                            { 6, -22.6078944 },
                            { 6.025, -22.5599748 },
                            { 6.05, -22.5122942 },
                            { 6.075, -22.4648515 },
                            { 6.1, -22.4176453 },
                            { 6.125, -22.3706746 },
                            { 6.15, -22.3239382 },
                            { 6.175, -22.2774349 },
                            { 6.2, -22.2311635 },
                            { 6.225, -22.1851228 },
                            { 6.25, -22.1393119 },
                            { 6.275, -22.0937293 },
                            { 6.3, -22.0483742 },
                            { 6.325, -22.0032452 },
                            { 6.35, -21.9583414 },
                            { 6.375, -21.9136614 },
                            { 6.4, -21.8692044 },
                            { 6.425, -21.824969 },
                            { 6.45, -21.7809543 },
                            { 6.475, -21.7371591 },
                            { 6.5, -21.6935824 },
                            { 6.525, -21.6502229 },
                            { 6.55, -21.6070798 },
                            { 6.575, -21.5641518 },
                            { 6.6, -21.5214379 },
                            { 6.625, -21.478937 },
                            { 6.65, -21.4366482 },
                            { 6.675, -21.3945702 },
                            { 6.7, -21.3527021 },
                            { 6.725, -21.3110428 },
                            { 6.75, -21.2695913 },
                            { 6.775, -21.2283466 },
                            { 6.8, -21.1873075 },
                            { 6.825, -21.1464732 },
                            { 6.85, -21.1058425 },
                            { 6.875, -21.0654144 },
                            { 6.9, -21.025188 },
                            { 6.925, -20.9851622 },
                            { 6.95, -20.945336 },
                            { 6.975, -20.9057085 },
                            { 7, -20.8662786 },
                            { 7.025, -20.8270454 },
                            { 7.05, -20.7880078 },
                            { 7.075, -20.749165 },
                            { 7.1, -20.7105159 },
                            { 7.125, -20.6720595 },
                            { 7.15, -20.6337949 },
                            { 7.175, -20.5957212 },
                            { 7.2, -20.5578374 },
                            { 7.225, -20.5201425 },
                            { 7.25, -20.4826357 },
                            { 7.275, -20.4453159 },
                            { 7.3, -20.4081822 },
                            { 7.325, -20.3712337 },
                            { 7.35, -20.3344696 },
                            { 7.375, -20.2978887 },
                            { 7.4, -20.2614904 },
                            { 7.425, -20.2252735 },
                            { 7.45, -20.1892373 },
                            { 7.475, -20.1533809 },
                            { 7.5, -20.1177032 },
                            { 7.525, -20.0822035 },
                            { 7.55, -20.0468809 },
                            { 7.575, -20.0117344 },
                            { 7.6, -19.9767633 },
                            { 7.625, -19.9419665 },
                            { 7.65, -19.9073433 },
                            { 7.675, -19.8728928 },
                            { 7.7, -19.8386141 },
                            { 7.725, -19.8045064 },
                            { 7.75, -19.7705687 },
                            { 7.775, -19.7368004 },
                            { 7.8, -19.7032005 },
                            { 7.825, -19.6697681 },
                            { 7.85, -19.6365025 },
                            { 7.875, -19.6034028 },
                            { 7.9, -19.5704682 },
                            { 7.925, -19.5376979 },
                            { 7.95, -19.5050909 },
                            { 7.975, -19.4726467 },
                            { 8, -19.4403642 },
                            { 8.025, -19.4082428 },
                            { 8.05, -19.3762815 },
                            { 8.075, -19.3444797 },
                            { 8.1, -19.3128365 },
                            { 8.125, -19.2813511 },
                            { 8.15, -19.2500227 },
                            { 8.175, -19.2188506 },
                            { 8.2, -19.1878339 },
                            { 8.225, -19.156972 },
                            { 8.25, -19.1262639 },
                            { 8.275, -19.0957091 },
                            { 8.3, -19.0653066 },
                            { 8.325, -19.0350558 },
                            { 8.35, -19.0049558 },
                            { 8.375, -18.9750059 },
                            { 8.4, -18.9452055 },
                            { 8.425, -18.9155536 },
                            { 8.45, -18.8860497 },
                            { 8.475, -18.8566929 },
                            { 8.5, -18.8274825 },
                            { 8.525, -18.7984178 },
                            { 8.55, -18.7694981 },
                            { 8.575, -18.7407226 },
                            { 8.6, -18.7120906 },
                            { 8.625, -18.6836015 },
                            { 8.65, -18.6552544 },
                            { 8.675, -18.6270487 },
                            { 8.7, -18.5989837 },
                            { 8.725, -18.5710586 },
                            { 8.75, -18.5432728 },
                            { 8.775, -18.5156257 },
                            { 8.8, -18.4881164 },
                            { 8.825, -18.4607443 },
                            { 8.85, -18.4335087 },
                            { 8.875, -18.406409 },
                            { 8.9, -18.3794444 },
                            { 8.925, -18.3526143 },
                            { 8.95, -18.325918 },
                            { 8.975, -18.2993549 },
                            { 9, -18.2729242 },
                            { 9.025, -18.2466254 },
                            { 9.05, -18.2204578 },
                            { 9.075, -18.1944206 },
                            { 9.1, -18.1685133 },
                            { 9.125, -18.1427353 },
                            { 9.15, -18.1170858 },
                            { 9.175, -18.0915642 },
                            { 9.2, -18.0661699 },
                            { 9.225, -18.0409023 },
                            { 9.25, -18.0157607 },
                            { 9.275, -17.9907445 },
                            { 9.3, -17.965853 },
                            { 9.325, -17.9410857 },
                            { 9.35, -17.916442 },
                            { 9.375, -17.8919211 },
                            { 9.4, -17.8675226 },
                            { 9.425, -17.8432457 },
                            { 9.45, -17.8190899 },
                            { 9.475, -17.7950546 },
                            { 9.5, -17.7711392 },
                            { 9.525, -17.747343 },
                            { 9.55, -17.7236655 },
                            { 9.575, -17.7001061 },
                            { 9.6, -17.6766643 },
                            { 9.625, -17.6533393 },
                            { 9.65, -17.6301307 },
                            { 9.675, -17.6070378 },
                            { 9.7, -17.5840601 },
                            { 9.725, -17.561197 },
                            { 9.75, -17.538448 },
                            { 9.775, -17.5158123 },
                            { 9.8, -17.4932896 },
                            { 9.825, -17.4708793 },
                            { 9.85, -17.4485807 },
                            { 9.875, -17.4263933 },
                            { 9.9, -17.4043165 },
                            { 9.925, -17.3823499 },
                            { 9.95, -17.3604929 },
                            { 9.975, -17.3387448 },};

    ASSERT_TRUE(testing::seq_eq(ums[{0, 0}], exp));
    ASSERT_TRUE(testing::seq_eq(ums[{0, 1}], exp));
    // gid == 1 is different, but of same size
    EXPECT_EQ((ums[{1, 0}].size()), exp.size());
    ASSERT_FALSE(testing::seq_eq(ums[{1, 0}], exp));
    // now check the spikes
    std::sort(spikes.begin(), spikes.end());
    EXPECT_EQ(spikes.size(), 3u);
    std::vector<double> sexp{2, 4, 5};
    ASSERT_TRUE(testing::seq_almost_eq<double>(spikes, sexp));
}
