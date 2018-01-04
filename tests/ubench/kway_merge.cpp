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
#include <iostream>

#include <benchmark/benchmark.h>

unsigned next_power_2(unsigned x) {
    unsigned n = 1;
    while (n<x) n<<=1;
    return n;
}

template <typename T>
struct tourney_tree {
    using iterator = typename std::vector<T>::const_iterator;
    using value_type = T;
    using key_val = std::pair<unsigned, iterator>;

    tourney_tree(const std::vector<std::vector<T>>& input):
        input_(input),
        n_lanes_(input_.size())
    {
        // must have at least 1 queue
        assert(n_lanes_!=0);
        // maximum value in unsigned limits how many queues we can have
        assert(n_lanes_<(1u<<(sizeof(unsigned)*8-1)));

        leaves_ = next_power_2(n_lanes_);
        nodes_ = 2*(leaves_-1)+1; // 2*l-1 witho overflow protection

        heap_.resize(nodes_);

        for (auto i=0u; i<leaves_; ++i) {
            heap_[leaf(i)] = i<n_lanes_?
                key_val(i, input[i].begin()):
                key_val(n_lanes_, iterator()); // invalid node
        }

        setup(0);
    }

    void setup(unsigned i) {
        if (is_leaf(i)) return;
        setup(left(i));
        setup(right(i));
        merge_up(i);
    };

    void merge_up(unsigned i) {
        const auto l = left(i);
        const auto r = right(i);
        heap_[i] = compare(l, r)? heap_[l]: heap_[r];
    }

    void update_lane(unsigned lane) {
        unsigned i = leaf(lane);

        // update the iterator and test if the lane is empty
        ++heap_[i].second;
        if (heap_[i].second==input_[lane].end()) {
            heap_[i].first = leaves_; // mark lane as empty
        }

        // walk from leaf to root
        while ((i=parent(i))) {
            merge_up(i);
        }
        merge_up(0); // handle to root
    }

    bool compare(unsigned l, unsigned r) {
        if (!is_valid(l)) {
            return false;
        }
        if (!is_valid(r)) {
            return true;
        }
        return *(heap_[l].second)<*(heap_[r].second);
    }

    unsigned parent(unsigned i) {
        return (i-1)>>1;
    }
    unsigned left(unsigned i) {
        return (i<<1) + 1;
    }
    unsigned right(unsigned i) {
        return left(i)+1;
    }
    unsigned leaf(unsigned i) {
        return i+leaves_-1;
    }
    bool is_leaf(unsigned i) {
        return i>=leaves_-1;
    }
    bool is_in_range(unsigned i) {
        return i<leaf(n_lanes_);
    }
    bool is_valid(unsigned i) {
        return heap_[i].first<n_lanes_;
        //auto lane = heap_[i].first;
        //return lane<n_lanes_ && heap_[i].second!=input_[lane].end();
    }

    void print() {
        for (auto i=0; i<nodes_; ++i) {
            if (is_valid(i))
                std::cout << *(heap_[i].second) << " ";
            else
                std::cout << "* ";
        }
        std::cout << "\n";
    }


    bool empty() {
        return !is_valid(0);
    }

    const T& head() {
        return *(heap_[0].second);
    }

    void pop() {
        auto lane = heap_[0].first;
        update_lane(lane);
    }

    std::vector<key_val> heap_;
    const std::vector<std::vector<T>>& input_;
    unsigned leaves_;
    unsigned nodes_;
    unsigned n_lanes_;
};


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

std::vector<float> kway_tourney(const std::vector<std::vector<float>>& lanes) {
    auto n = 0u;
    for (auto& l: lanes) n+=l.size();

    std::vector<float> m;
    m.reserve(n);

    tourney_tree<float> tree(lanes);

    while (!tree.empty()) {
        m.push_back(tree.head());
        tree.pop();
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
    unsigned n = state.range(1);

    // generate input information
    std::vector<std::vector<float>> lanes(num_lanes);
    auto seed = 0u;
    for (auto& l: lanes) {
        std::mt19937 gen(seed++);
        std::uniform_int_distribution<unsigned> dis_n(n/2, n);
        auto m = dis_n(gen);
        l.reserve(m);
        std::uniform_real_distribution<> dis(0, 1e6);
        for (auto i=0u; i<m; ++i) {
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
void test_tourney (benchmark::State& state) { test(state, kway_tourney);  }


void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto num_lanes: {2, 3, 5, 7, 10, 100, 1000}) {
        for (auto n: {100, 1000, 10000, 100000}) {
            b->Args({num_lanes, n});
        }
    }
}

//BENCHMARK(test_dumb)->Apply(run_custom_arguments);
BENCHMARK(test_naive)->Apply(run_custom_arguments);
BENCHMARK(test_heap)->Apply(run_custom_arguments);
BENCHMARK(test_tourney)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
