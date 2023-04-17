#include <random>
#include <unordered_map>
#include <vector>

#include <benchmark/benchmark.h>

#include "util/tourney_tree.hpp"
#include "merge_events.hpp"

#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

constexpr auto T = 1000.0; // ms

using rndgen = std::mt19937_64;

struct payload {
    payload(std::size_t ncells, std::size_t ev_per_cell) {
        auto dt = T/ev_per_cell;
        for(auto cell = 0ull; cell < ncells; ++cell) {
            auto gen = arb::poisson_schedule(1/dt, rndgen{cell});
            auto times = gen.events(0, T);
            evts.emplace_back();
            auto& evt = evts.back();
            for (auto t: arb::util::make_range(times)) {
                evt.emplace_back(arb::spike_event{42, t, 0.23});
                ++size;
            }
            span.emplace_back(arb::util::make_range(evt.data(), evt.data() + evt.size()));
        }
    }

    std::vector<std::vector<arb::spike_event>> evts;
    std::vector<arb::event_span> span;
    std::size_t size = 0;
};

static void BM_tree(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    const payload data{ncells, ev_per_cell};

    while (state.KeepRunning()) {
        arb::pse_vector out;
        // Need to do this here, normally the wrapper does this
        out.reserve(data.size);
        auto tmp = data.span;
        arb::tree_merge_events(tmp, out);
        benchmark::ClobberMemory();
    }
}

static void BM_linear(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    const payload data{ncells, ev_per_cell};

    while (state.KeepRunning()) {
        arb::pse_vector out;
        // Need to do this here, normally the wrapper does this
        out.reserve(data.size);
        // Clone the input, merge clobbers it.
        auto tmp = data.span;
        arb::linear_merge_events(tmp, out);
        benchmark::ClobberMemory();
    }
}

static void BM_queue(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    const payload data{ncells, ev_per_cell};

    while (state.KeepRunning()) {
        arb::pse_vector out;
        // Need to do this here, normally the wrapper does this
        out.reserve(data.size);
        // Clone the input, merge clobbers it.
        auto tmp = data.span;
        arb::pqueue_merge_events(tmp, out);
        benchmark::ClobberMemory();
    }
}

static void BM_default(benchmark::State& state) {
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    const payload data{ncells, ev_per_cell};

    while (state.KeepRunning()) {
        arb::pse_vector out;
        // NOTE: This wrapper _does_ do the allocation.
        // Clone the input, merge clobbers it.
        auto tmp = data.span;
        arb::merge_events(tmp, out);
        benchmark::ClobberMemory();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncells: {5, 13, 23, 41, 53}) {
        for (auto ev_per_cell: {8, 32, 256, 1024}) {
            b->Args({ncells, ev_per_cell});
        }
    }
}

BENCHMARK(BM_tree)->Apply(run_custom_arguments);
BENCHMARK(BM_linear)->Apply(run_custom_arguments);
BENCHMARK(BM_queue)->Apply(run_custom_arguments);
BENCHMARK(BM_default)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
