#include "../gtest.h"

#include <backends/gpu_intrinsics.hpp>
#include <memory/managed_ptr.hpp>

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
}

// test atomic addition wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_add) {
    int expected = (128*129)/2;

    auto f = nest::mc::memory::make_managed_ptr<float>(0.f);
    kernels::test_atomic_add<<<1, 128>>>(f.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(float(expected), *f);

    auto d = nest::mc::memory::make_managed_ptr<double>(0.);
    kernels::test_atomic_add<<<1, 128>>>(d.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(double(expected), *d);
}

// test atomic subtraction wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_sub) {
    int expected = -(128*129)/2;

    auto f = nest::mc::memory::make_managed_ptr<float>(0.f);
    kernels::test_atomic_sub<<<1, 128>>>(f.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(float(expected), *f);

    auto d = nest::mc::memory::make_managed_ptr<double>(0.);
    kernels::test_atomic_sub<<<1, 128>>>(d.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(double(expected), *d);
}

