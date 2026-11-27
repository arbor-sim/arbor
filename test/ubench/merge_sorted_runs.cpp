// Merging the sorted runs of a level of the neighbourhood spike exchange.
//
// A level buffers one run of spikes per epoch, each sorted by source, back to
// back in one vector, and merges them before sending. Compare the pairwise
// merge used by neighborhood_router with a pairwise std::inplace_merge, a linear
// k-way merge and a priority queue, as chosen between in merge_events, and with
// sorting.

#include <algorithm>
#include <queue>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <arbor/spike.hpp>

#include "communication/neighborhood_router.hpp"

namespace {

using arb::spike;

bool by_source(const spike& a, const spike& b) { return a.source < b.source; }

struct payload {
    std::vector<spike> spikes;
    std::vector<std::size_t> runs;
};

// k runs of n spikes each from 10000 sources, each run sorted by source.
payload make_runs(std::size_t k, std::size_t n) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<arb::cell_gid_type> gid(0, 9999);
    payload p;
    for (std::size_t r = 0; r < k; ++r) {
        p.runs.push_back(p.spikes.size());
        auto first = p.spikes.size();
        for (std::size_t i = 0; i < n; ++i) p.spikes.push_back(spike({gid(rng), 0}, double(r)));
        std::sort(p.spikes.begin() + first, p.spikes.end(), by_source);
    }
    return p;
}

// As in neighborhood_router: pairwise std::merge between two buffers.
void pairwise(std::vector<spike>& spikes, const std::vector<std::size_t>& runs, std::vector<spike>& buffer) {
    arb::merge_sorted_runs(spikes, runs, buffer);
}

// Pairwise std::inplace_merge, needing no second buffer.
void pairwise_inplace(std::vector<spike>& spikes, const std::vector<std::size_t>& in_runs, std::vector<spike>&) {
    auto runs = in_runs;
    runs.push_back(spikes.size());
    while (runs.size() > 2) {
        std::vector<std::size_t> merged{runs.front()};
        for (std::size_t i = 0; i + 2 < runs.size(); i += 2) {
            std::inplace_merge(spikes.begin() + runs[i], spikes.begin() + runs[i + 1], spikes.begin() + runs[i + 2], by_source);
            merged.push_back(runs[i + 2]);
        }
        if (runs.size() % 2 == 0) merged.push_back(runs.back());
        runs = std::move(merged);
    }
}

// As linear_merge_events: take the minimum of all run fronts.
void linear(std::vector<spike>& spikes, const std::vector<std::size_t>& runs, std::vector<spike>& out) {
    std::vector<std::size_t> front(runs), end(runs.begin() + 1, runs.end());
    end.push_back(spikes.size());
    out.clear();
    for (;;) {
        std::size_t best = runs.size();
        for (std::size_t r = 0; r < runs.size(); ++r) {
            if (front[r] < end[r] && (best == runs.size() || by_source(spikes[front[r]], spikes[front[best]]))) best = r;
        }
        if (best == runs.size()) break;
        out.push_back(spikes[front[best]++]);
    }
    std::swap(spikes, out);
}

// As pqueue_merge_events: a min-heap of the run fronts.
void heap(std::vector<spike>& spikes, const std::vector<std::size_t>& runs, std::vector<spike>& out) {
    std::vector<std::size_t> end(runs.begin() + 1, runs.end());
    end.push_back(spikes.size());
    using entry = std::pair<std::size_t, std::size_t>; // position, run
    auto later = [&](const entry& a, const entry& b) { return by_source(spikes[b.first], spikes[a.first]); };
    std::priority_queue<entry, std::vector<entry>, decltype(later)> q(later);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (runs[r] < end[r]) q.push({runs[r], r});
    }
    out.clear();
    while (!q.empty()) {
        auto [pos, r] = q.top();
        q.pop();
        out.push_back(spikes[pos]);
        if (++pos < end[r]) q.push({pos, r});
    }
    std::swap(spikes, out);
}

void stable_sort(std::vector<spike>& spikes, const std::vector<std::size_t>&, std::vector<spike>&) {
    std::stable_sort(spikes.begin(), spikes.end(), by_source);
}

template <void (*merge)(std::vector<spike>&, const std::vector<std::size_t>&, std::vector<spike>&)>
void bench(benchmark::State& state) {
    const auto data = make_runs(state.range(0), state.range(1));
    std::vector<spike> spikes, buffer;
    buffer.reserve(data.spikes.size());
    for (auto _: state) {
        // Every variant starts from a copy, since the merges work in place.
        spikes = data.spikes;
        merge(spikes, data.runs, buffer);
        benchmark::DoNotOptimize(spikes.data());
        benchmark::ClobberMemory();
    }
    if (!std::is_sorted(spikes.begin(), spikes.end(), by_source)) state.SkipWithError("not sorted");
    state.SetItemsProcessed(state.iterations()*data.spikes.size());
}

// Runs: number of epochs buffered. Spikes per run: per epoch and rank.
void arguments(benchmark::internal::Benchmark* b) {
    for (long k: {2, 4, 8, 16, 32, 64, 128, 256}) {
        for (long n: {16, 256, 4096}) b->Args({k, n});
    }
    b->ArgNames({"runs", "spikes_per_run"});
}

} // namespace

BENCHMARK(bench<pairwise>)->Name("pairwise")->Apply(arguments);
BENCHMARK(bench<pairwise_inplace>)->Name("pairwise_inplace")->Apply(arguments);
BENCHMARK(bench<linear>)->Name("linear")->Apply(arguments);
BENCHMARK(bench<heap>)->Name("heap")->Apply(arguments);
BENCHMARK(bench<stable_sort>)->Name("stable_sort")->Apply(arguments);

BENCHMARK_MAIN();
