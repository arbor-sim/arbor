/* Compare implementations of the test:
 *    ∃i: a[i]<b[i]?
 * for device-side arrays a and b.
 *
 * Four implementations are compared:
 * 1. Copy both vectors to host, compare there.
 * 2. Use the thrust library.
 * 3. Use a small custom cuda kernel.
 * 4. Same custom cuda kernel with once-off allocation of return value.
 */

// Explicitly undef NDEBUG for assert below.
#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <cuda_profiler_api.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>

template <typename Impl>
void bench_generic(benchmark::State& state, const Impl& impl) {
    std::size_t n = state.range(0);

    int oop = state.range(1);
    double p_inclusion = oop? 1.0/oop: 0;

    // Make device vectors `a` and `b` of size n, and with 
    // `p_inclusion` chance of `a[i]` < `b[i]` for any given `i`.

    std::bernoulli_distribution B(p_inclusion);
    std::uniform_int_distribution<int> U(0, 99);
    std::minstd_rand R;

    std::vector<int> xs(n);
    std::generate(xs.begin(), xs.end(), [&]() { return U(R); });

    thrust::device_vector<int> a(n);
    thrust::copy(xs.begin(), xs.end(), a.begin());

    bool differ = false;
    thrust::device_vector<int> b(n);
    for (std::size_t i = 0; i<n; ++i) {
        if (B(R)) {
            xs[i] += 3;
            differ = true;
        }
    }
    thrust::copy(xs.begin(), xs.end(), b.begin());
    cudaDeviceSynchronize();

    // Run benchmark.

    bool result = false;
    int* aptr = thrust::raw_pointer_cast(a.data());
    int* bptr = thrust::raw_pointer_cast(b.data());

    cudaProfilerStart();
    while (state.KeepRunning()) {
        result = impl(n, aptr, bptr);
        benchmark::DoNotOptimize(result);
    }
    cudaProfilerStop();

    // validate result
    assert(result==differ);
}

bool host_copy_compare(std::size_t n, int* aptr, int* bptr) {
    std::vector<int> a(n);
    std::vector<int> b(n);

    cudaMemcpy(a.data(), aptr, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), bptr, n*sizeof(int), cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i<n; ++i) {
        if (a[i]<b[i]) return true;
    }
    return false;
}

struct thrust_cmp_pred {
    __device__
    bool operator()(const thrust::tuple<int, int>& p) const {
        return thrust::get<0>(p) < thrust::get<1>(p);
    }
};

bool thrust_compare(std::size_t n, int* aptr, int* bptr) {
    thrust::device_ptr<int> a(aptr), b(bptr);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(a, b));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(a+n, b+n));

    return thrust::any_of(zip_begin, zip_end, thrust_cmp_pred{});
}

__global__ void cuda_cmp_kernel(std::size_t n, int* aptr, int* bptr, int* rptr) {
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int cmp = i<n? aptr[i]<bptr[i]: 0;
    if (__syncthreads_or(cmp)) *rptr=1;
}

bool custom_cuda_compare(std::size_t n, int* aptr, int* bptr) {
    int result;

    constexpr int blockwidth = 128;
    int nblock = n? 1+(n-1)/blockwidth: 0;

    void* rptr;
    cudaMalloc(&rptr, sizeof(int));
    cudaMemset(rptr, 0, sizeof(int));
    cuda_cmp_kernel<<<nblock, blockwidth>>>(n, aptr, bptr, (int *)rptr);
    cudaMemcpy(&result, rptr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(rptr);

    return (bool)result;
}

template <typename T>
struct device_store {
    device_store() {
    void* p;
    cudaMalloc(&p, sizeof(T));
    ptr = (T*)p;
    }

    ~device_store() { if (ptr) cudaFree(ptr); }

    device_store(const device_store&&) = delete;
    device_store(device_store&&) = delete;

    T* get() { return ptr; }

private:
    T* ptr = nullptr;
};

bool custom_cuda_compare_noalloc(std::size_t n, int* aptr, int* bptr) {
    static thread_local device_store<int> state;
    int result;

    constexpr int blockwidth = 128;
    int nblock = n? 1+(n-1)/blockwidth: 0;

    cudaMemset(state.get(), 0, sizeof(int));
    cuda_cmp_kernel<<<nblock, blockwidth>>>(n, aptr, bptr, state.get());
    cudaMemcpy(&result, state.get(), sizeof(int), cudaMemcpyDeviceToHost);

    return (bool)result;
}

void bench_host_copy_compare(benchmark::State& state) {
    bench_generic(state, host_copy_compare);
}

void bench_thrust_compare(benchmark::State& state) {
    bench_generic(state, thrust_compare);
}

void bench_custom_cuda_compare(benchmark::State& state) {
    bench_generic(state, custom_cuda_compare);
}

void bench_custom_cuda_compare_noalloc(benchmark::State& state) {
    bench_generic(state, custom_cuda_compare_noalloc);
}

// Run benches over 256 to circa 1e6 elements, with three cases:
//
// Arg1       Values
// ---------------------------------------------------
//    0       `a` and `b` equal
//  200       `a` and `b` differ circa 1 in 200 places.
//    4       `a` and `b` differ circa 1 in 4 places.

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (int n=1<<8; n<=1<<20; n*=2) {
        for (int oop: {0, 200, 4}) {
            // Uncomment to set fixed iteration count (for e.g. profiling):
            //b->Iterations(20);
            b->Args({n, oop});
        }
    }
}

BENCHMARK(bench_host_copy_compare)->Apply(run_custom_arguments);
BENCHMARK(bench_thrust_compare)->Apply(run_custom_arguments);
BENCHMARK(bench_custom_cuda_compare)->Apply(run_custom_arguments);
BENCHMARK(bench_custom_cuda_compare_noalloc)->Apply(run_custom_arguments);

BENCHMARK_MAIN();
