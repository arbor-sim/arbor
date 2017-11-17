// Compare methods for performing the "event-setup" step for mc cell groups.
// The key concern is how to take an unsorted set of events
//
// TODO: We assume that the cells in a cell group are numbered contiguously,
// i.e. 0:ncells-1. The cells in an mc_cell_group are not typically thus,
// instead a hash table is used to look up the cell_group local index from the
// gid. A similar lookup should be added to theses tests, to more accurately
// reflect the mc_cell_group implementation.
//
// TODO: The staged_events output is a vector of postsynaptic_spike_event, not
// a deliverable event.

#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

std::vector<float> kway_dumb(const std::vector<std::vector<float>>& lanes) {
    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    std::vector<float> m;
    m.reserve(n);

    for (auto& l: lanes) {
        m.insert(m.begin(), std::begin(l), std::end(l));
    }
    std::sort(m.begin(), m.end());

    return m;
}

std::vector<float> kway_naive(const std::vector<std::vector<float>>& lanes) {
    const unsigned k = lanes.size();
    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    std::vector<float> m;
    m.reserve(n);

    using p = std::pair<unsigned, std::vector<float>::const_iterator>;
    std::vector<p> tops;
    tops.reserve(k);
    for (auto i=0u; i<k; ++i) {
        auto& l = lanes[i];
        if (l.size()) {
            tops.push_back({i, l.begin()});
        }
    }

    while (!tops.empty()) {
        auto it = std::min_element(
                tops.begin(), tops.end(),
                [](p l, p r) {return *(l.second)<*(r.second);});
        m.push_back(*(it->second));
        ++(it->second);
        if (it->second==lanes[it->first].end()) {
            tops.erase(it);
        }
    }

    return m;
}

std::vector<float> kway_heap(const std::vector<std::vector<float>>& lanes) {
    using p = std::pair<float, unsigned>;
    auto op = [](p l, p r) {return l.first>r.first;};

    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    const unsigned k = lanes.size();

    if (lanes.size()==1) return lanes[0];

    std::vector<p> heap;
    heap.reserve(k);

    std::vector<float> m;
    m.reserve(n);

    // Build heap with first entry in each lane, and make list of iterators that
    // tracks the entry in each lane that is currently in the queue.
    std::vector<std::vector<float>::const_iterator> tops;
    tops.reserve(k);
    for (auto& l: lanes) {
        tops.push_back(l.begin());
        if (l.size()) {
            heap.push_back({l.front(), tops.size()-1});
            ++tops.back();
        }
    }
    std::make_heap(heap.begin(), heap.end(), op);

    while (!heap.empty()) {
        const auto l = heap.front().second;
        m.push_back(heap.front().first);
        std::pop_heap(heap.begin(), heap.end(), op);
        heap.pop_back();
        if (tops[l]!=lanes[l].end()) {
            heap.push_back({*tops[l], l});
            std::push_heap(heap.begin(), heap.end(), op);
            ++tops[l];
        }
    }

    return m;
}

template <typename F>
void test(benchmark::State& state, F f) {
    const unsigned num_lanes = state.range(0);
    const unsigned n = state.range(1);

    // generate input information
    std::vector<std::vector<float>> lanes(num_lanes);
    auto seed = 0u;
    for (auto& l: lanes) {
        l.reserve(n);
        std::mt19937 gen(seed++);
        std::uniform_real_distribution<> dis(0, 1e6);
        for (auto i=0u; i<n; ++i) {
            l.push_back(dis(gen));
        }
        std::sort(l.begin(), l.end());
    }

    // run the benchmark
    while (state.KeepRunning()) {
        f(lanes);
        benchmark::ClobberMemory();
    }
}

void test_dumb (benchmark::State& state) { test(state, kway_dumb);  }
void test_naive(benchmark::State& state) { test(state, kway_naive); }
void test_heap (benchmark::State& state) { test(state, kway_heap);  }


void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto num_lanes: {3, 10, 20, 100}) {
        for (auto n: {100, 1000, 10000}) {
            b->Args({num_lanes, n});
        }
    }
}

BENCHMARK(test_dumb)->Apply(run_custom_arguments);
BENCHMARK(test_naive)->Apply(run_custom_arguments);
BENCHMARK(test_heap)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
