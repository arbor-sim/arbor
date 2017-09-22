// Compare methods for performing the "event-setup" step for mc cell groups.
// The key concern is how to take an unsorted set of events
//
// TODO: These benchmarks assume that the cells in a cell group are continguous,
// numbered 0:ncells-1. The cells in an mc_cell_group are not typically thus,
// instead a hash table is used to look up the cell_group local index from the
// gid. A similar lookup should be added to theses tests, to moare accurately
// reflect the mc_cell_group implementation.
//
// TODO: The staged_events output is a vector of postsynaptic_spike_event, not
// a deliverable event.

#include <random>
#include <vector>

#include <event_binner.hpp>
#include <event_queue.hpp>
#include <backends/event.hpp>

#include <benchmark/benchmark.h>

using namespace nest::mc;

std::vector<postsynaptic_spike_event> generate_inputs(size_t ncells, size_t ev_per_cell) {
    std::vector<postsynaptic_spike_event> input_events;
    std::default_random_engine engine;
    std::uniform_int_distribution<cell_gid_type>(0u, ncells);

    std::default_random_engine gen;
    std::uniform_int_distribution<cell_gid_type>
        gid_dist(0u, ncells-1);
    std::uniform_real_distribution<float>
        time_dist(0.f, 1.f);

    input_events.reserve(ncells*ev_per_cell);
    for (std::size_t i=0; i<ncells*ev_per_cell; ++i) {
        postsynaptic_spike_event ev;
        auto gid = gid_dist(gen);
        auto t = time_dist(gen);
        ev.target = {cell_gid_type(gid), cell_lid_type(0)};
        ev.time = t;
        ev.weight = 0;
        input_events.push_back(ev);
    }

    return input_events;
}

void single_queue(benchmark::State& state) {
    using pev = postsynaptic_spike_event;

    const std::size_t ev_per_cell = state.range(0);
    const std::size_t ncells = state.range(1);

    std::vector<pev> input_events = generate_inputs(ncells, ev_per_cell);

    //auto binner = event_binner(binning_kind::regular, 0.001);
    event_queue<pev> events;
    while (state.KeepRunning()) {
        // push events into a single queue
        for (const auto& e: input_events) {
            events.push(e);
        }

        // pop from queue to form single sorted vector
        std::vector<pev> staged_events;
        staged_events.reserve(events.size());
        while (auto e = events.pop_if_before(1.f)) {
            staged_events.push_back(*e);
        }
        // sort the staged events in order of target id
        std::stable_sort(
            staged_events.begin(), staged_events.end(),
            [](const pev& l, const pev& r) {return l.target.gid<r.target.gid;});

        // TODO: calculate the partition ranges.

        /*
        std::vector<deliverable_event> staged_events;
        staged_events.reserve(events.size());
        while (auto e = events.pop_if_before(2.f)) {
            staged_events.emplace_back(
                binner.bin(e->target.gid, e->time, 0.f),
                // TODO: this should do a lookup in a hash table?
                //get_target_handle(e->target),
                target_handle(0, 0, e->target.gid),
                e->weight);
        }

        binner.reset();
        */

        benchmark::ClobberMemory();
    }
}

void n_queue(benchmark::State& state) {
    using pev = postsynaptic_spike_event;
    const std::size_t ev_per_cell = state.range(0);
    const std::size_t ncells = state.range(1);

    auto input_events = generate_inputs(ncells, ev_per_cell);

    // state
    std::vector<event_queue<pev>> event_lanes(ncells);
    std::vector<size_t> part(ncells+1);

    while (state.KeepRunning()) {
        part[0] = 0;

        // push events into the queue corresponding to target cell
        for (const auto& e: input_events) {
            event_lanes[e.target.gid].push(e);
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

        benchmark::ClobberMemory();
    }
}

void n_vector(benchmark::State& state) {
    using pev = postsynaptic_spike_event;
    const size_t ev_per_cell = state.range(0);
    const size_t ncells = state.range(1);

    auto input_events = generate_inputs(ncells, ev_per_cell);

    // state
    std::vector<std::vector<pev>> event_lanes(ncells);
    std::vector<size_t> part(ncells+1);
    std::vector<decltype(input_events.begin())> ext(ncells);

    struct ev_lt_pred {
        bool operator()(float t, const pev& ev) { return t<ev.time; }
        bool operator()(const pev& ev, float t) { return ev.time<t; }
    };

    //auto binner = event_binner(binning_kind::regular, 0.001);
    while (state.KeepRunning()) {
        ext.clear();

        // push events into a single queue
        for (const auto& e: input_events) {
            event_lanes[e.target.gid].push_back(e);
        }
        for (auto& lane: event_lanes) {
            std::sort(lane.begin(), lane.end(),
                      [](const pev& l, const pev& r) {return l.time<r.time;});
            ext.push_back(
                std::lower_bound(lane.begin(), lane.end(), 1.f, ev_lt_pred()));
        }
        part[0] = 0;
        for (size_t i=0; i<ncells; ++i) {
            part[i+1] = part[i] + std::distance(event_lanes[i].begin(), ext[i]);
        }
        std::vector<postsynaptic_spike_event> staged_events(part.back());
        auto b = staged_events.begin();
        for (size_t i=0; i<ncells; ++i) {
            std::copy(event_lanes[i].begin(), ext[i], b+part[i]);
        }

        auto i=0u;
        for (auto& lane: event_lanes) {
            lane.erase(lane.begin(), ext[i++]);
        }

        benchmark::ClobberMemory();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncells: {1, 10, 100, 1000, 10000}) {
        //for (auto ev_per_cell: {128, 256, 512, 1024, 2048, 4096}) {
        for (auto ev_per_cell: {256, 1024, 4096}) {
            b->Args({ncells, ev_per_cell});
        }
    }
}

//BENCHMARK(run_original)->Apply(run_custom_arguments);
BENCHMARK(single_queue)->Apply(run_custom_arguments);
BENCHMARK(n_queue)->Apply(run_custom_arguments);
BENCHMARK(n_vector)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
