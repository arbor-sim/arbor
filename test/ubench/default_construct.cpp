// Compare value- vs default- initialized vector performance.

// Explicitly undef NDEBUG for assert below.
#undef NDEBUG

#include <cassert>
#include <vector>

#include <benchmark/benchmark.h>

#include "util/span.hpp"

using arb::util::make_span;

template <typename Allocator>
struct default_construct_adaptor: Allocator {
private:
    using traits = typename std::allocator_traits<Allocator>;

public:
    using pointer = typename traits::pointer;
    using value_type = typename traits::value_type;

    default_construct_adaptor() noexcept {}

    template <typename... Args>
    default_construct_adaptor(Args&&... args):
        Allocator(std::forward<Args...>(args...))
    {}


    template <typename U>
    default_construct_adaptor(const default_construct_adaptor<U>& b) noexcept: Allocator(b) {}

    void construct(pointer p) {
        ::new (static_cast<void*>(p)) value_type;
    }

    template <typename... Args>
    void construct(pointer p, Args&&... args) {
        ::new (static_cast<void*>(p)) value_type(std::forward<Args...>(args)...);
    }

    template <typename U>
    struct rebind {
        using other = default_construct_adaptor<typename traits::template rebind_alloc<U>>;
    };

};


template <typename Container>
unsigned run_accumulate(std::size_t n) {
    Container c(n);

    unsigned s = 0;
    for (unsigned& x: c) {
        x = ++s;
    }
    s = 0;
    for (unsigned x: c) {
        s += x;
    }
    return s;
}

template <typename Container>
unsigned run_accumulate_range_init(std::size_t n) {
    auto values = make_span(1, n+1);
    Container c(values.begin(), values.end());

    unsigned s = 0;
    for (unsigned x: c) {
        s += x;
    }
    return s;
}

template <unsigned (*Fn)(std::size_t)>
void bench_container(benchmark::State& state) {
    std::size_t n = state.range(0);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(Fn(n));
    }

    // check!
    unsigned s = (n*(n+1))/2;
    assert(s==Fn(n));
}

template <typename T>
using dc_vector = std::vector<T, default_construct_adaptor<std::allocator<T>>>;

auto bench_vector = bench_container<run_accumulate<std::vector<unsigned>>>;
auto bench_dc_vector = bench_container<run_accumulate<dc_vector<unsigned>>>;

auto bench_vector_range = bench_container<run_accumulate_range_init<std::vector<unsigned>>>;
auto bench_dc_vector_range = bench_container<run_accumulate_range_init<dc_vector<unsigned>>>;


BENCHMARK(bench_vector)->Range(1<<10, 1<<20);
BENCHMARK(bench_dc_vector)->Range(1<<10, 1<<20);

BENCHMARK(bench_vector_range)->Range(1<<10, 1<<20);
BENCHMARK(bench_dc_vector_range)->Range(1<<10, 1<<20);

BENCHMARK_MAIN();

