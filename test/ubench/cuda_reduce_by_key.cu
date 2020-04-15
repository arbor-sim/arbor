// Compare implementations of reduce by key
//   u[key[i]] += v[i]
// where key is sorted in ascending order and may contain repeated keys

// Explicitly undef NDEBUG for assert below.
#undef NDEBUG

#include <iostream>
#include <vector>
#include <random>

#include <benchmark/benchmark.h>

#include <memory/memory.hpp>
#include <backends/gpu/kernels/reduce_by_key.hpp>
#include <backends/gpu/intrinsics.hpp>
#include <util/rangeutil.hpp>

using namespace arb;

// Run benchmarks
//  * with between 100:1million entries to update
//  * with between 1:1000 threads updating each entry
// This corresponds to a range from a single 100 compartment cell with 1 synapse
// per compartment to 100k cells with 10 compartments and 10k synapses each.

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto n_comp: {100, 1000, 10000, 100000, 1000000}) {
        for (auto syn_per_comp: {1, 10, 100, 1000}) {
            b->Args({n_comp, syn_per_comp});
        }
    }
}

template <typename T, typename I>
__global__
void reduce_by_shuffle(const T* src, T* dst, const I* index, int n) {
    unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid<n) {
        gpu::reduce_by_key(src[tid], dst, index[tid]);
    }
}

template <typename T, typename I>
__global__
void reduce_by_atomic(const T* src, T* dst, const I* index, int n) {
    unsigned tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid<n) {
        gpu_atomic_add(dst + index[tid], src[tid]);
    }
}

template <typename T, typename Impl>
void bench_runner(benchmark::State& state, const Impl& impl, T) {
    int n_cmp = state.range(0);
    int n = n_cmp*state.range(1);

    // find a uniform random assignment of indices to compartments
    std::uniform_int_distribution<int> U(0, n_cmp-1);
    std::minstd_rand R;

    std::vector<unsigned> hist(n_cmp);
    for (int i=0; i<n; ++i) {
        ++hist[U(R)];
    }

    std::vector<int> index;
    index.reserve(n);
    for (int i=0; i<n_cmp; ++i) {
        index.insert(index.end(), hist[i], i);
    }

    using array = memory::device_vector<T>;
    using iarray = memory::device_vector<int>;

    // copy inputs to the device
    array out(n_cmp, T(0));
    array in(n, T(1));
    iarray idx = memory::make_const_view(index);

    // set up cuda events for custom timer of kernel executation times
    cudaEvent_t start_e;
    cudaEvent_t stop_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&stop_e);
    while (state.KeepRunning()) {
        int block_dim = 128;
        int grid_dim = (n-1)/block_dim + 1;

        cudaEventRecord(start_e, 0);
        // call the kernel
        impl<<<grid_dim, block_dim>>>(in.data(), out.data(), idx.data(), n);
        cudaEventRecord(stop_e, 0);

        // wait for kernel call to finish before querying the time taken
        cudaEventSynchronize(stop_e);
        float time_taken = 0.0f;
        cudaEventElapsedTime(&time_taken, start_e, stop_e);

        // pass custom time to benchmark framework
        state.SetIterationTime(time_taken);
    }
    cudaEventDestroy(start_e);
    cudaEventDestroy(stop_e);
}

// runners for single precision tests
void shuffle_32(benchmark::State& state) {
    bench_runner(state, reduce_by_shuffle<float, int>, float());
}
void atomic_32(benchmark::State& state) {
    bench_runner(state, reduce_by_atomic<float, int>, float());
}
BENCHMARK(shuffle_32)->Apply(run_custom_arguments)->UseManualTime();
BENCHMARK(atomic_32)->Apply(run_custom_arguments)->UseManualTime();

// runners for double precision tests
void shuffle_64(benchmark::State& state) {
    bench_runner(state, reduce_by_shuffle<double, int>, double());
}
void atomic_64(benchmark::State& state) {
    bench_runner(state, reduce_by_atomic<double, int>, double());
}
BENCHMARK(shuffle_64)->Apply(run_custom_arguments)->UseManualTime();
BENCHMARK(atomic_64)->Apply(run_custom_arguments)->UseManualTime();

BENCHMARK_MAIN();
