// Test performance of vectorization for mechanism implementations.
//
// Start with pas (passive dendrite) mechanism

#include <chrono>
#include <iostream>
#include <thread>

#include <arbor/version.hpp>

#include "threading/threading.hpp"

#include <benchmark/benchmark.h>

using namespace arb;

void run(unsigned long us_per_task, unsigned tasks, threading::task_system* ts) {
    auto duration = std::chrono::microseconds(us_per_task);
    arb::threading::parallel_for::apply(
            0, tasks, ts,
            [&](unsigned i){std::this_thread::sleep_for(duration);});
}

void run_nested(unsigned long us_per_task, unsigned tasks, threading::task_system* ts) {
    auto duration = std::chrono::microseconds(us_per_task);
    arb::threading::parallel_for::apply(0, tasks, ts, [&](unsigned i) {
       arb::threading::parallel_for::apply(0, 1, ts, [&](unsigned i) {
          std::this_thread::sleep_for(duration);});});
}

void task_test_nested(benchmark::State& state) {
    const unsigned us_per_task = state.range(0);
    arb::threading::task_system ts;
    const auto nthreads = ts.get_num_threads();
    const unsigned total_us = 1000000;
    const unsigned num_tasks = nthreads*total_us/us_per_task;

    while (state.KeepRunning()) {
        run_nested(us_per_task, num_tasks, &ts);
    }
}

void task_test(benchmark::State& state) {
    const unsigned us_per_task = state.range(0);
    arb::threading::task_system ts;
    const auto nthreads = ts.get_num_threads();
    const unsigned total_us = 1000000;
    const unsigned num_tasks = nthreads*total_us/us_per_task;

    while (state.KeepRunning()) {
        run(us_per_task, num_tasks, &ts);
    }
}

void us_per_task(benchmark::internal::Benchmark *b) {
    for (auto us_per_task: {10, 100, 250, 500, 1000, 10000}) {
        b->Args({us_per_task});
    }
}

BENCHMARK(task_test)->Apply(us_per_task);
BENCHMARK(task_test_nested)->Apply(us_per_task);
BENCHMARK_MAIN();
