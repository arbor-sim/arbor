#include <algorithm>
#include <random>
#include <vector>
#include <algorithm>

#include <benchmark/benchmark.h>

#include <arbor/spike_event.hpp>
#include <arbor/spike.hpp>
#include <arbor/common_types.hpp>

#include "connection.hpp"
#include "ankerl/unordered_dense.h"

// Emulate spike queuing for a cable cell group.
// - receive an order list of spike events
// - and a mapping of target cell to a queue
// - to dispatch a resulting to a single queue per cell
// NOTE there are some subleties:
// 1. events are always used in the group due to the way MPI communication is set up
// 2. events might be used multiple times


// Generate M events per each of N cellsm allowing repetition of the source
// NOTE we satisfy sublety 1 by generating sources in the same range as the targets.
auto generate_events(size_t n_cell, size_t evt_per_cell) {
    std::default_random_engine engine;
    std::mt19937 gen;
    std::uniform_int_distribution<arb::cell_gid_type>(0u, n_cell);
    std::uniform_int_distribution<arb::cell_gid_type> gid_dist(0u, n_cell - 1);
    std::uniform_real_distribution<arb::time_type> time_dist(0.f, 1.f);

    std::vector<arb::spike> res;
    res.reserve(n_cell*evt_per_cell);
    for (std::size_t ix = 0; ix < n_cell*evt_per_cell; ++ix) {
        res.push_back({
                          arb::cell_member_type { .gid=gid_dist(gen), .index=0},
                          time_dist(gen),
                      });
    }
    std::sort(res.begin(), res.end());
    return res;
}

auto generate_connections(size_t n_cell, size_t conn_per_cell) {
    std::default_random_engine engine;
    std::mt19937 gen;
    std::uniform_int_distribution<arb::cell_gid_type>(0u, n_cell);
    std::uniform_int_distribution<arb::cell_gid_type> gid_dist(0u, n_cell - 1);
    std::uniform_real_distribution<float> real_dist(0.f, 1.f);

    std::vector<arb::connection> res;
    res.reserve(n_cell*conn_per_cell);
    for (std::size_t ix = 0; ix < n_cell*conn_per_cell; ++ix) {
        res.push_back(
                      arb::connection {
                          .source={ .gid=gid_dist(gen), .index=0},
                          .target=0,
                          .weight=real_dist(gen),
                          .delay=real_dist(gen),
                          .index_on_domain=gid_dist(gen),
                      });
    }
    std::sort(res.begin(), res.end());
    return res;
}

void naive(benchmark::State& state) {
    size_t n_cell        = state.range(0);
    size_t conn_per_cell = state.range(1);
    size_t evt_per_cell  = state.range(2);

    auto conns  = generate_connections(n_cell, conn_per_cell);
    auto spikes = generate_events(n_cell, evt_per_cell);

    auto out   = std::vector<arb::pse_vector>(n_cell);

    while (state.KeepRunning()) {
        for (auto& spike: spikes) {
            for (auto& conn: conns) {
                if (conn.source == spike.source) {
                    out[conn.index_on_domain].emplace_back(conn.target, spike.time + conn.delay, conn.weight);
                }
            }
        }

        for (auto& q: out) q.clear();
    }
}

void sorted_conn_spike(benchmark::State& state) {
    size_t n_cell        = state.range(0);
    size_t conn_per_cell = state.range(1);
    size_t evt_per_cell  = state.range(2);

    auto conns  = generate_connections(n_cell, conn_per_cell);
    auto spikes = generate_events(n_cell, evt_per_cell);

    auto n_conn  = conns.size();
    auto n_spike = spikes.size();

    auto out   = std::vector<arb::pse_vector>(n_cell);

    while (state.KeepRunning()) {
        size_t sidx = 0;
        size_t cidx = 0;
        while (sidx < n_spike && cidx < n_conn) {
            auto src = conns[cidx].source;
            while (sidx < n_spike && spikes[sidx].source < src) ++sidx;
            if (sidx >= n_spike) continue;
            auto fst = sidx;
            for (; cidx < n_conn && conns[cidx].source == src; ++cidx) {
                const auto& conn = conns[cidx];
                auto& queue = out[conn.index_on_domain];
                // Handle all connections with the same source
                // scan the range of spikes, once per connection
                for (sidx= fst; sidx < n_spike && spikes[sidx].source == src; ++sidx) {
                    queue.emplace_back(conn.target, spikes[sidx].time + conn.delay, conn.weight);
                }
            }
            // once we leave here, sp will be at the end of the eglible range
            // and all connections with the same source will have been treated.
            // so, we can just leave sp at this end.
        }

        for (auto& q: out) q.clear();
    }
}

void binary_search(benchmark::State& state) {
    size_t n_cell        = state.range(0);
    size_t conn_per_cell = state.range(1);
    size_t evt_per_cell  = state.range(2);

    auto conns  = generate_connections(n_cell, conn_per_cell);
    auto spikes = generate_events(n_cell, evt_per_cell);

    auto out   = std::vector<arb::pse_vector>(n_cell);

    while (state.KeepRunning()) {
        for (const auto& spike: spikes) {
            auto src = spike.source;
            auto it = std::lower_bound(conns.begin(), conns.end(),
                                       src,
                                       [] (const auto& it, const auto& src) { return it.source < src; });
            while ((it != conns.end()) && (it->source == src)) {
                const auto& conn = *it;
                auto& queue = out[conn.index_on_domain];
                queue.emplace_back(conn.target, spike.time + conn.delay, conn.weight);
                ++it;
            }
        }

        for (auto& q: out) q.clear();
    }
}

auto src_to_key (const arb::cell_member_type& m) { return std::bit_cast<uint64_t>(m); }

struct aos {
    std::vector<arb::cell_size_type> idx_on_domain;
    std::vector<uint64_t> srcs;
    std::vector<arb::cell_lid_type> dests;
    std::vector<float> weights;
    std::vector<float> delays;
};

aos generate_aos(const std::vector<arb::connection>& conns) {
    auto res = aos{};
    for (const auto& con: conns) {
        auto source = src_to_key(con.source);
        res.idx_on_domain.push_back(con.index_on_domain);
        res.srcs.push_back(source);
        res.dests.push_back(con.target);
        res.weights.push_back(con.weight);
        res.delays.push_back(con.delay);
    }
    return res;
}

void binary_search_aos(benchmark::State& state) {
    size_t n_cell        = state.range(0);
    size_t conn_per_cell = state.range(1);
    size_t evt_per_cell  = state.range(2);

    auto conns  = generate_connections(n_cell, conn_per_cell);
    auto spikes = generate_events(n_cell, evt_per_cell);
    auto cons   = generate_aos(conns);
    auto queues = std::vector<arb::pse_vector>(n_cell);

    while (state.KeepRunning()) {
        auto cbeg = cons.srcs.begin();
        auto ccur = cbeg;
        auto cend = cons.srcs.end();
        auto clen = std::distance(cbeg, cend);

        auto send = spikes.end();
        auto scur  = spikes.begin();
        while (scur < send) {
            auto src = scur->source;
            auto source = src_to_key(src);
            auto ctmp = std::lower_bound(ccur, cend, source);
            // We now longer need to search below the current source; they are sorted
            ccur = ctmp;
            // Start creation of events. This can (likely: will) create more
            // than one event per incoming spike as multiple connections
            // exist for one source.
            // Remember the starting point of the run of spikes with the same source
            auto stmp = scur;
            // Iterate connections from the same source
            for (auto idx = std::distance(cbeg, ctmp); (idx < clen) && (cons.srcs[idx] == source); ++idx) {
                auto iod    = cons.idx_on_domain[idx];
                auto dest   = cons.dests[idx];
                auto delay  = cons.delays[idx];
                auto weight = cons.weights[idx];
                auto& queue = queues[iod];
                // Make events for all spikes with the same source
                for(scur = stmp; (scur < send) && (scur->source == src); ++scur) {
                    queue.emplace_back(dest, scur->time + delay, weight);
                }
                // NOTE: Without the reset `scur = stmp` the cursor `scur` will
                //       be (correctly) at the end of the range.
                // NOTE: For the same reason will step the connection cursor `ccur`
                ++ccur;
            }
        }
        for (auto& q: queues) q.clear();
    }
}

void hashtable_aos(benchmark::State& state) {
    size_t n_cell        = state.range(0);
    size_t conn_per_cell = state.range(1);
    size_t evt_per_cell  = state.range(2);

    auto conns  = generate_connections(n_cell, conn_per_cell);
    auto spikes = generate_events(n_cell, evt_per_cell);
    auto aos    = generate_aos(conns);

    ankerl::unordered_dense::map<uint64_t, std::pair<size_t, size_t>> first_occurence;

    for (size_t idx = 0; idx < aos.srcs.size(); ++idx) {
        const auto& source = aos.srcs[idx];
        if (!first_occurence.contains(source)) first_occurence.emplace(source, std::make_pair(idx, 0));
        first_occurence[source].second += 1;
    }

    auto out = std::vector<arb::pse_vector>(n_cell);

    while (state.KeepRunning()) {
        auto send = spikes.end();
        auto sit = spikes.begin();
        for (; sit < send; ++sit) {
            auto src = sit->source;
            auto source = src_to_key(src);
            const auto& [fst, len] = first_occurence.at(source);
            for (; (sit < send) && (sit->source == src); ++sit) {
                for (auto idx = fst; idx < fst + len; ++idx) {
                    auto& queue = out[aos.idx_on_domain[idx]];
                    queue.emplace_back(aos.dests[idx], sit->time + aos.delays[idx], aos.weights[idx]);
                }
            }
        }

        for (auto& q: out) q.clear();
    }
}

void run_custom_arguments(::benchmark::Benchmark* b) {
    for (auto n_cell: {10, 1000, 10000}) {
        for (auto conn_per_cell: {64, 128, 256}) {
            for (auto evt_per_cell: {64, 128, 256}) {
                b->Args({n_cell, conn_per_cell, evt_per_cell});
            }
        }
    }
}

// BENCHMARK(naive)->Apply(run_custom_arguments);
// BENCHMARK(sorted_conn_spike)->Apply(run_custom_arguments);
// BENCHMARK(binary_search)->Apply(run_custom_arguments);
BENCHMARK(binary_search_aos)->Apply(run_custom_arguments);
BENCHMARK(hashtable_aos)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
