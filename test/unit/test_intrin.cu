#include "../gtest.h"

#include <limits>

#include "backends/gpu/cuda_atomic.hpp"
#include "backends/gpu/math_cu.hpp"
#include "memory/memory.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace kernels {
    template <typename T>
    __global__
    void test_atomic_add(T* x) {
        cuda_atomic_add(x, threadIdx.x+1);
    }

    template <typename T>
    __global__
    void test_atomic_sub(T* x) {
        cuda_atomic_sub(x, threadIdx.x+1);
    }

    __global__
    void test_min(double* x, double* y, double* result) {
        const auto i = threadIdx.x;
        result[i] = arb::gpu::min(x[i], y[i]);
    }

    __global__
    void test_max(double* x, double* y, double* result) {
        const auto i = threadIdx.x;
        result[i] = arb::gpu::max(x[i], y[i]);
    }

    __global__
    void test_exprelr(double* x, double* result) {
        const auto i = threadIdx.x;
        result[i] = arb::gpu::exprelr(x[i]);
    }

}

// test atomic addition wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_add) {
    int expected = (128*129)/2;

    arb::memory::device_vector<float> f(1);
    f[0] = 0.f;

    kernels::test_atomic_add<<<1, 128>>>(f.data());

    EXPECT_EQ(float(expected), f[0]);

    arb::memory::device_vector<double> d(1);
    d[0] = 0.f;

    kernels::test_atomic_add<<<1, 128>>>(d.data());

    EXPECT_EQ(double(expected), d[0]);
}

// test atomic subtraction wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_sub) {
    int expected = -(128*129)/2;

    arb::memory::device_vector<float> f(1);
    f[0] = 0.f;

    kernels::test_atomic_sub<<<1, 128>>>(f.data());

    EXPECT_EQ(float(expected), f[0]);

    arb::memory::device_vector<double> d(1);
    d[0] = 0.f;

    kernels::test_atomic_sub<<<1, 128>>>(d.data());

    EXPECT_EQ(double(expected), d[0]);
}

TEST(gpu_intrinsics, minmax) {
    const double inf = std::numeric_limits<double>::infinity();
    struct X {
        double lhs;
        double rhs;
        double expected_min;
        double expected_max;
    };

    std::vector<X> inputs = {
        {  0,    1,    0,   1},
        { -1,    1,   -1,   1},
        { 42,   42,   42,  42},
        {inf, -inf, -inf, inf},
        {  0,  inf,    0, inf},
        {  0, -inf, -inf,   0},
    };

    const auto n = arb::util::size(inputs);

    arb::memory::device_vector<double> lhs(n);
    arb::memory::device_vector<double> rhs(n);
    arb::memory::device_vector<double> result(n);

    using arb::util::make_span;

    for (auto i: make_span(0, n)) {
        lhs[i] = inputs[i].lhs;
        rhs[i] = inputs[i].rhs;
    }

    // test min
    kernels::test_min<<<1, n>>>(lhs.data(), rhs.data(), result.data());
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_min);
    }

    kernels::test_min<<<1, n>>>(rhs.data(), lhs.data(), result.data());
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_min);
    }

    // test max
    kernels::test_max<<<1, n>>>(lhs.data(), rhs.data(), result.data());
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_max);
    }

    kernels::test_max<<<1, n>>>(rhs.data(), lhs.data(), result.data());
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_max);
    }
}

TEST(gpu_intrinsics, exprelr) {
    constexpr double dmin = std::numeric_limits<double>::min();
    constexpr double dmax = std::numeric_limits<double>::max();
    constexpr double deps = std::numeric_limits<double>::epsilon();
    double inputs[] = {-1.,  -0.,  0.,  1., -dmax,  -dmin,  dmin,  dmax, -deps, deps, 10*deps, 100*deps, 1000*deps};

    auto n = arb::util::size(inputs);
    arb::memory::device_vector<double> x(arb::memory::host_view<double>(inputs, n));
    arb::memory::device_vector<double> result(n);

    kernels::test_exprelr<<<1,n>>>(x.data(), result.data());

    auto index = arb::util::make_span(0, n);
    for (auto i: index) {
        auto x = inputs[i];
        double expected = std::fabs(x)<deps? 1.0: x/std::expm1(x);
        double error = std::fabs(expected-double(result[i]));
        double relerr = expected==0.? error: error/std::fabs(expected);
        EXPECT_TRUE(relerr<deps);
    }
}
