// Compare methods for performing the "event-setup" step for mc cell groups.
// The key concern is how to take an unsorted set of events
//
// TODO: We assume that the cells in a cell group are numbered contiguously,
// i.e. 0:ncells-1. The cells in an mc_cell_group are not typically thus,
// instead a hash table is used to look up the cell_group local index from the
// gid. A similar lookup should be added to theses tests, to more accurately
// reflect the mc_cell_group implementation.
//
// TODO: The staged_events output is a vector of spike_event, not
// a deliverable event.

#include <algorithm>
#include <random>
#include <vector>
#include <algorithm>

#include <benchmark/benchmark.h>

#include "event_queue.hpp"
#include "backends/event.hpp"

using namespace arb;

std::vector<std::pair<cell_gid_type, spike_event>> generate_inputs(size_t ncells, size_t ev_per_cell) {
    std::vector<std::pair<cell_gid_type, spike_event>> input_events;
    std::default_random_engine engine;
    std::uniform_int_distribution<cell_gid_type>(0u, ncells);

    std::mt19937 gen;
    std::uniform_int_distribution<cell_gid_type>
        gid_dist(0u, ncells-1);
    std::uniform_real_distribution<float>
        time_dist(0.f, 1.f);

    input_events.reserve(ncells*ev_per_cell);
    for (std::size_t i=0; i<ncells*ev_per_cell; ++i) {
        spike_event ev;
        auto gid = gid_dist(gen);
        auto t = time_dist(gen);
        ev.target = cell_lid_type(0);
        ev.time = t;
        ev.weight = 0;
        input_events.emplace_back(gid, ev);
    }

    return input_events;
}

void n_queue(benchmark::State& state) {
    using pev = spike_event;
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    auto input_events = generate_inputs(ncells, ev_per_cell);

    // state
    std::vector<event_queue<pev>> event_lanes(ncells);
    std::vector<size_t> part(ncells+1);

    while (state.KeepRunning()) {
        part[0] = 0;

        // push events into the queue corresponding to target cell
        for (const auto& e: input_events) {
            event_lanes[e.first].push(e.second);
        }

        // pop from queue to form single sorted vector
        std::vector<pev> staged_events;
        staged_events.reserve(input_events.size());
        size_t i=0;
        for (auto& lane: event_lanes) {
            while (auto e = lane.pop_if_before(1.f)) {
                staged_events.push_back(*e);
            }
            part[++i] = staged_events.size();
        }

        // clobber lanes for the next round of benchmarking
        for (auto& lane: event_lanes) {
            lane.clear();
        }

        benchmark::ClobberMemory();
    }
}

void n_vector(benchmark::State& state) {
    using pev = spike_event;
    const std::size_t ncells = state.range(0);
    const std::size_t ev_per_cell = state.range(1);

    auto input_events = generate_inputs(ncells, ev_per_cell);

    // state
    std::vector<std::vector<pev>> event_lanes(ncells);
    std::vector<size_t> part(ncells+1);
    std::vector<size_t> ext(ncells);

    struct ev_lt_pred {
        bool operator()(float t, const pev& ev) { return t<ev.time; }
        bool operator()(const pev& ev, float t) { return ev.time<t; }
    };

    // NOTE: this is a "full" implementation, that can handle the case where
    // input_events contains events that are to be delivered after the current
    // delivery interval. The event_lanes vectors keep undelivered events.
    while (state.KeepRunning()) {
        ext.clear();

        // push events into a per-cell vectors (unsorted)
        for (const auto& e: input_events) {
            event_lanes[e.first].push_back(e.second);
        }
        // sort each per-cell queue and keep track of the subset of sorted
        // events that are to be delivered in this interval.
        for (auto& lane: event_lanes) {
            std::sort(lane.begin(), lane.end(),
                      [](const pev& l, const pev& r) {return l.time<r.time;});
            ext.push_back(
                std::distance(
                    lane.begin(),
                    std::lower_bound(lane.begin(), lane.end(), 1.f, ev_lt_pred())));
        }
        // calculate partition of output buffer according to target cell gid
        part[0] = 0;
        for (size_t i=0; i<ncells; ++i) {
            part[i+1] = part[i] + ext[i];
        }
        // copy events into the output flat buffer
        std::vector<spike_event> staged_events(part.back());
        auto b = staged_events.begin();
        for (size_t i=0; i<ncells; ++i) {
            auto bi = event_lanes[i].begin();
            std::copy(bi, bi+ext[i], b+part[i]);
        }

        // remove events that were delivered from the event lanes
        auto i=0u;
        for (auto& lane: event_lanes) {
            auto b = lane.begin();
            lane.erase(b, b+ext[i++]);
        }

        // clobber contents of lane for next round of benchmark
        for (auto& lane: event_lanes) {
            lane.clear();
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

BENCHMARK(n_queue)->Apply(run_custom_arguments);
BENCHMARK(n_vector)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
