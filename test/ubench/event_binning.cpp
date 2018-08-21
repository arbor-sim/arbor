// Test overheads of hashing function using in binning of events (following method).
// Only tests hashing overhead, which isn't enough for proper analysis of costs.
//
// Keep this test as a prototype for testing, esp. when looking into binning.

#include <random>
#include <unordered_map>
#include <vector>

#include <benchmark/benchmark.h>

#include <arbor/spike_event.hpp>

#include "event_queue.hpp"
#include "backends/event.hpp"


using namespace arb;

std::vector<cell_gid_type> generate_gids(size_t n) {
    std::mt19937 engine;
    std::uniform_int_distribution<cell_gid_type> dist(1u, 3u);

    std::vector<cell_gid_type> gids;
    gids.reserve(n);

    gids.push_back(0);
    while (gids.size()<n) {
        gids.push_back(gids.back()+dist(engine));
    }

    return gids;
}

std::vector<pse_vector> generate_inputs(const std::vector<cell_gid_type>& gids, size_t ev_per_cell) {
    auto ncells = gids.size();
    std::vector<pse_vector> input_events;

    std::uniform_int_distribution<cell_gid_type>(0u, ncells);
    std::mt19937 gen;
    std::uniform_int_distribution<cell_gid_type>
        gid_dist(0u, ncells-1);

    input_events.resize(ncells);
    for (std::size_t i=0; i<ncells*ev_per_cell; ++i) {
        spike_event ev;
        auto idx = gid_dist(gen);
        auto gid = gids[idx];
        auto t = 1.;
        ev.target = {cell_gid_type(gid), cell_lid_type(0)};
        ev.time = t;
        ev.weight = 0;
        input_events[idx].push_back(ev);
    }

    return input_events;
}

void no_hash(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    // state
    auto gids = generate_gids(ncells);
    auto input_events = generate_inputs(gids, ev_per_cell);

    std::vector<float> times(ncells);
    while (state.KeepRunning()) {
        for (size_t i=0; i<ncells; ++i) {
            for (auto& ev: input_events[i]) {
                times[i] += ev.time;
            }
        }

        benchmark::ClobberMemory();
    }
}

void yes_hash(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    // state
    auto gids = generate_gids(ncells);
    auto input_events = generate_inputs(gids, ev_per_cell);

    std::unordered_map<cell_gid_type, float> times;
    for (auto gid: gids) {
        times[gid] = 0;
    }
    while (state.KeepRunning()) {
        for (size_t i=0; i<ncells; ++i) {
            for (auto& ev: input_events[i]) {
                times[i] += ev.time;
            }
        }

        benchmark::ClobberMemory();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncells: {1, 10, 100, 1000, 10000}) {
        for (auto ev_per_cell: {128, 256, 512, 1024, 2048, 4096}) {
            b->Args({ncells, ev_per_cell});
        }
    }
}

BENCHMARK(no_hash)->Apply(run_custom_arguments);
BENCHMARK(yes_hash)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
