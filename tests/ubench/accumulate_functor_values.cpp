#undef NDEBUG

#include <cassert>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>

#include <util/span.hpp>
#include <util/transform.hpp>


using namespace nest::mc;

inline long long square(long long x) { return x*x; }

long long sum_squares_to(long long x) {
    return x<1? 0: (2*x*x*x+3*x*x+x)/6;
}

template <typename Func>
void partial_sums_direct(Func f, int upto, std::vector<long long>& psum) {
    long long sum = 0;
    for (int i=1; i<=upto; ++i) {
        sum += f(i);
        psum[i-1] = sum;
    }
}

template <typename Func>
void partial_sums_transform(Func f, int upto, std::vector<long long>& psum) {
    auto nums = util::span<long long>(1, upto+1);
    auto values = util::transform_view(nums, f);
    std::partial_sum(values.begin(), values.end(), psum.begin());
}

void bench_accum_direct(benchmark::State& state) {
    int upto = state.range(0);
    std::vector<long long> psum(upto);

    while (state.KeepRunning()) {
        partial_sums_direct(square, upto, psum);
    }

    for (int i = 0; i<upto; ++i) {
        assert(sum_squares_to(i+1)==psum[i]);
    }
}

void bench_accum_transform(benchmark::State& state) {
    int upto = state.range(0);
    std::vector<long long> psum(upto);

    while (state.KeepRunning()) {
        partial_sums_transform(square, upto, psum);
    }

    for (int i = 0; i<upto; ++i) {
        assert(sum_squares_to(i+1)==psum[i]);
    }
}


BENCHMARK(bench_accum_direct)->RangeMultiplier(2)->Range(2, 1<<10);
BENCHMARK(bench_accum_transform)->RangeMultiplier(2)->Range(2, 1<<10);
BENCHMARK_MAIN();

