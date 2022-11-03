#include <gtest/gtest.h>

#include <random>
#include <vector>
#include <any>

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_source_cell.hpp>

#include "util/rangeutil.hpp"
#include "util/transform.hpp"

#include "common.hpp"
using namespace arb;

struct play_spikes: public recipe {
    play_spikes(std::vector<schedule> spike_times): spike_times_(std::move(spike_times)) {}

    cell_size_type num_cells() const override { return spike_times_.size(); }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::spike_source; }
    util::unique_any get_cell_description(cell_gid_type gid) const override {
        return spike_source_cell("src", spike_times_.at(gid));
    }

    std::vector<schedule> spike_times_;
};

static auto n_thread_context(unsigned n_thread) {
    return make_context(proc_allocation(std::max((int)n_thread, 1), -1));
}

struct null_recipe: arb::recipe {
    arb::cable_cell_global_properties properties;
    null_recipe() { properties.default_parameters = arb::neuron_parameter_defaults; }
    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override { return {arb::cable_cell{}}; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override { return {}; }
    std::any get_global_properties(arb::cell_kind) const override { return properties; }
};

// Test that cell models with empty morpologies build and run without error.
TEST(simulation, null) {
    auto r = null_recipe{};
    auto c = arb::make_context();
    auto d = arb::partition_load_balance(r, c);
    auto s = arb::simulation(r, c, d);
    s.run(0.05, 0.01);
}

// Test with simulation builder
TEST(simulation, null_builder) {
    auto r = null_recipe{};
    {
        arb::simulation s = arb::simulation::create(r);
        s.run(0.05, 0.01);
    }
    {
        arb::simulation s = arb::simulation::create(r).set_seed(42);
        s.run(0.05, 0.01);
    }
    {
        auto c = arb::make_context();
        arb::simulation s = arb::simulation::create(r).set_context(c);
        s.run(0.05, 0.01);
    }
    {
        auto c = arb::make_context();
        auto d = arb::partition_load_balance(r, c);
        arb::simulation s = arb::simulation::create(r).set_context(c).set_decomposition(d);
        s.run(0.05, 0.01);
    }
}

TEST(simulation, spike_global_callback) {
    constexpr unsigned n = 5;
    double t_max = 10.;

    std::vector<schedule> spike_times;
    for (unsigned i = 0; i<n; ++i) {
        spike_times.push_back(poisson_schedule(0., 20./t_max, std::minstd_rand(1000+i)));
    }

    std::vector<spike> expected_spikes;
    for (unsigned i = 0; i<n; ++i) {
        util::append(expected_spikes, util::transform_view(util::make_range(spike_times.at(i).events(0, t_max)),
            [i](double time) { return spike({i, 0}, time); }));
        spike_times.at(i).reset();
    }

    play_spikes rec(spike_times);
    auto ctx = n_thread_context(4);
    auto decomp = partition_load_balance(rec, ctx);
    simulation sim(rec, ctx, decomp);

    std::vector<spike> collected;
    sim.set_global_spike_callback([&](const std::vector<spike>& spikes) {
        collected.insert(collected.end(), spikes.begin(), spikes.end());
    });

    double tfinal = 0.7*t_max;
    constexpr double dt = 0.01;
    sim.run(tfinal, dt);

    auto spike_lt = [](spike a, spike b) { return a.time<b.time || (a.time==b.time && a.source<b.source); };
    std::sort(expected_spikes.begin(), expected_spikes.end(), spike_lt);
    std::sort(collected.begin(), collected.end(), spike_lt);

    auto cut_from = std::partition_point(expected_spikes.begin(), expected_spikes.end(), [tfinal](auto spike) { return spike.time<tfinal; });
    expected_spikes.erase(cut_from, expected_spikes.end());
    EXPECT_EQ(expected_spikes, collected);
}

struct lif_chain: public recipe {
    lif_chain(unsigned n, double delay, schedule triggers):
        n_(n), delay_(delay), triggers_(std::move(triggers)) {}

    cell_size_type num_cells() const override { return n_; }

    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override {
        // A hair-trigger LIF cell with tiny time constant and no refractory period.
        lif_cell lif("src", "tgt");
        lif.tau_m = 0.01;           // time constant (ms)
        lif.t_ref = 0;              // refactory period (ms)
        lif.V_th = lif.E_L + 0.001; // threshold voltage 1 µV higher than resting
        return lif;
    }

    std::vector<cell_connection> connections_on(cell_gid_type target) const override {
        if (target) {
            return {cell_connection({target-1, "src"}, {"tgt"}, weight_, delay_)};
        }
        else {
            return {};
        }
    }

    std::vector<event_generator> event_generators(cell_gid_type target) const override {
        if (target) {
            return {};
        }
        else {
            return {event_generator({"tgt"}, weight_, triggers_)};
        }
    }

    static constexpr double weight_ = 2.0;
    unsigned n_;
    double delay_;
    schedule triggers_;
};

TEST(simulation, restart) {
    std::vector<double> trigger_times = {1., 2., 3.};
    double delay = 10;
    unsigned n = 5;
    lif_chain rec(n, delay, explicit_schedule(trigger_times));

    // Expect spike times to be almost exactly according to trigger times,
    // plus delays along the chain of cells.

    std::vector<spike> expected_spikes;
    for (auto t: trigger_times) {
        for (unsigned i = 0; i<n; ++i) {
            expected_spikes.push_back(spike({i, 0}, i*delay+t));
        }
    }

    auto spike_lt = [](spike a, spike b) { return a.time<b.time || (a.time==b.time && a.source<b.source); };
    std::sort(expected_spikes.begin(), expected_spikes.end(), spike_lt);

    auto ctx = n_thread_context(4);
    auto decomp = partition_load_balance(rec, ctx);
    simulation sim(rec, ctx, decomp);

    std::vector<spike> collected;
    sim.set_global_spike_callback([&](const std::vector<spike>& spikes) {
        collected.insert(collected.end(), spikes.begin(), spikes.end());
    });

    double tfinal = trigger_times.back()+delay*(n/2+0.1);
    constexpr double dt = 0.01;

    auto cut_from = std::partition_point(expected_spikes.begin(), expected_spikes.end(), [tfinal](auto spike) { return spike.time<tfinal; });
    expected_spikes.erase(cut_from, expected_spikes.end());

    // Run simulation in various numbers of stages, ranging from a single stage
    // to running it in stages of duration less than delay/2.

    for (double run_time = 0.1*delay; run_time<=tfinal; run_time *= 1.5) {
        SCOPED_TRACE(run_time);
        collected.clear();

        sim.reset();
        double t = 0;
        do {
            double run_to = std::min(tfinal, t + run_time);
            t = sim.run(run_to, dt);
            ASSERT_EQ(t, run_to);
        } while (t<tfinal);

        ASSERT_EQ(expected_spikes.size(), collected.size());
        for (unsigned i = 0; i<expected_spikes.size(); ++i) {
            EXPECT_EQ(expected_spikes[i].source, collected[i].source);
            EXPECT_DOUBLE_EQ(expected_spikes[i].time, collected[i].time);
        }
    }
}
