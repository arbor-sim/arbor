// Compare implementations of partial summation of the f(i) for i=1..n,
// for a simple square function.

// Explicitly undef NDEBUG for assert below.
#undef NDEBUG

#include <cassert>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>

#include "util/span.hpp"
#include "util/transform.hpp"

#define NOINLINE __attribute__((noinline))

using namespace arb;

inline long long square_function(long long x) { return x*x; }

struct square_object {
    long long operator()(long long x) const { return x*x; }
};

using result_vec = std::vector<long long>;

template <typename Func>
void partial_sums_direct(Func f, int upto, result_vec& psum) {
    long long sum = 0;
    for (int i=1; i<=upto; ++i) {
        sum += f(i);
        psum[i-1] = sum;
    }
}

template <typename Func>
void partial_sums_transform(Func f, int upto, result_vec& psum) {
    auto nums = util::span<long long>(1, upto+1);
    auto values = util::transform_view(nums, f);
    std::partial_sum(values.begin(), values.end(), psum.begin());
}

template <typename Impl>
void bench_generic(benchmark::State& state, const Impl& impl) {
    int upto = state.range(0);
    result_vec psum(upto);

    while (state.KeepRunning()) {
        impl(upto, psum);
        benchmark::ClobberMemory();
    }

    // validate result
    auto sum_squares_to = [](long long x) {return (2*x*x*x+3*x*x+x)/6; };
    for (int i = 0; i<upto; ++i) {
        assert(sum_squares_to(i+1)==psum[i]);
    }
}

void accum_direct_function(benchmark::State& state) {
    bench_generic(state,
        [](int upto, result_vec& psum) { partial_sums_direct(square_function, upto, psum); });
}

void accum_direct_object(benchmark::State& state) {
    bench_generic(state,
        [](int upto, result_vec& psum) { partial_sums_direct(square_object{}, upto, psum); });
}

void accum_transform_function(benchmark::State& state) {
    bench_generic(state,
        [](int upto, result_vec& psum) { partial_sums_transform(square_function, upto, psum); });
}

void accum_transform_object(benchmark::State& state) {
    bench_generic(state,
        [](int upto, result_vec& psum) { partial_sums_transform(square_object{}, upto, psum); });
}

BENCHMARK(accum_direct_function)->Range(64, 1024);
BENCHMARK(accum_transform_function)->Range(64, 1024);
BENCHMARK(accum_direct_object)->Range(64, 1024);
BENCHMARK(accum_transform_object)->Range(64, 1024);

BENCHMARK_MAIN();

