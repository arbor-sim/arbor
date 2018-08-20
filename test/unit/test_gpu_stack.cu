#include "../gtest.h"

#include <backends/gpu/stack.hpp>
#include <backends/gpu/stack_cu.hpp>
#include <backends/gpu/managed_ptr.hpp>
#include <arbor/execution_context.hpp>

using namespace arb;

TEST(stack, construction) {
    using T = int;

    execution_context context;
    gpu::stack<T> s(10, context.gpu);

    EXPECT_EQ(0u, s.size());
    EXPECT_EQ(10u, s.capacity());
}

// kernel and functors for testing push_back functionality
namespace kernels {
    template <typename F>
    __global__
    void push_back(gpu::stack_storage<int>& s, F f) {
        if (f(threadIdx.x)) {
            arb::gpu::push_back(s, int(threadIdx.x));
        }
    }

    struct all_ftor {
        __host__ __device__
        bool operator() (int i) {
            return true;
        }
    };

    struct even_ftor {
        __host__ __device__
        bool operator() (int i) {
            return (i%2)==0;
        }
    };

    struct odd_ftor {
        __host__ __device__
        bool operator() (int i) {
            return i%2;
        }
    };
}

TEST(stack, push_back) {
    using T = int;
    using stack = gpu::stack<T>;

    execution_context context;

    const unsigned n = 10;
    EXPECT_TRUE(n%2 == 0); // require n is even for tests to work
    auto s = stack(n, context.gpu);
    auto& sstorage = s.storage();

    kernels::push_back<<<1, n>>>(sstorage, kernels::all_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n, s.size());
    for (auto i=0; i<int(s.size()); ++i) {
        EXPECT_EQ(i, s[i]);
    }

    s.clear();
    kernels::push_back<<<1, n>>>(sstorage, kernels::even_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n/2, s.size());
    for (auto i=0; i<int(s.size())/2; ++i) {
        EXPECT_EQ(2*i, s[i]);
    }

    s.clear();
    kernels::push_back<<<1, n>>>(sstorage, kernels::odd_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n/2, s.size());
    for (auto i=0; i<int(s.size())/2; ++i) {
        EXPECT_EQ(2*i+1, s[i]);
    }
}

TEST(stack, overflow) {
    using T = int;
    using stack = gpu::stack<T>;

    execution_context context;

    const unsigned n = 10;
    auto s = stack(n, context.gpu);
    auto& sstorage = s.storage();
    EXPECT_FALSE(s.overflow());

    // push 2n items into a stack of size n
    kernels::push_back<<<1, 2*n>>>(sstorage, kernels::all_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n, s.size());
    EXPECT_EQ(2*n, s.pushes());
    EXPECT_TRUE(s.overflow());
}

TEST(stack, empty) {
    using T = int;
    using stack = gpu::stack<T>;

    execution_context context;

    stack s(0u, context.gpu);

    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.capacity(), 0u);

    EXPECT_EQ(s.storage().data, nullptr);
}
