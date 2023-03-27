#include <random>
#include <unordered_map>
#include <vector>

#include <benchmark/benchmark.h>

#include "util/tourney_tree.hpp"
#include "merge_events.hpp"

#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

constexpr auto T = 1000.0; // ms

struct payload {
    payload(ncells, ev_per_cell) {
        auto dt = T/events_per_cell;
        for(auto cell = 0ull; cell < ncells; ++cell) {
            auto sched = arb::poisson_schedule(1/dt, std::mt19937_64(cell));


        }
    }

    std::vector<std::vector<arb::spike_event>> evts;
    std::vector<arb::event_span> span;
};

static void BM_tree(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    payload data{ncells, ev_per_cell};

    while (state.KeepRunning()) {
        arb::pse_vector out;
        arb::tree_merge_events(span, out);
        benchmark::ClobberMemory();
    }
}

static void BM_linear(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);
    const int kind = state.range(2);

    auto gens = make_generator(ncells, ev_per_cell, kind);
    auto evts = std::vector<std::vector<arb::spike_event>>{};
    auto span = std::vector<arb::event_span>{};
    for (auto& gen: gens) {
        auto times = gen.events(0, T);
        evts.emplace_back();
        auto& evt = evts.back();
        for (auto t: arb::util::make_range(times)) {
            evt.emplace_back(arb::spike_event{42, t, 0.23});
        }
        span.emplace_back(arb::util::make_range(evt.data(), evt.data() + evt.size()));
    }
    while (state.KeepRunning()) {
        arb::pse_vector out;
        arb::merge_events(span, out);
        benchmark::ClobberMemory();
    }
}

static void BM_default(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);
    const int kind = state.range(2);

    auto gens = make_generator(ncells, ev_per_cell, kind);
    auto evts = std::vector<std::vector<arb::spike_event>>{};
    auto span = std::vector<arb::event_span>{};
    for (auto& gen: gens) {
        auto times = gen.events(0, T);
        evts.emplace_back();
        auto& evt = evts.back();
        for (auto t: arb::util::make_range(times)) {
            evt.emplace_back(arb::spike_event{42, t, 0.23});
        }
        span.emplace_back(arb::util::make_range(evt.data(), evt.data() + evt.size()));
    }
    while (state.KeepRunning()) {
        arb::pse_vector out;
        arb::merge_events(span, out);
        benchmark::ClobberMemory();
    }
}


void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (const auto& dist: {gen_regular, gen_poisson}) {
        for (auto ncells: {10, 100, 1000}) {
            for (auto ev_per_cell: {8, 32, 256}) {
                b->Args({ncells, ev_per_cell, dist});
            }
        }
    }
}

BENCHMARK(BM_tree)->Apply(run_custom_arguments);
BENCHMARK(BM_linear)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
