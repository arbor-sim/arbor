#include "../gtest.h"

#include <backends/gpu/stack.hpp>
#include <backends/gpu/managed_ptr.hpp>

using namespace nest::mc;

TEST(stack, construction) {
    using T = int;

    gpu::stack<T> s(10);

    EXPECT_EQ(0u, s.size());
    EXPECT_EQ(10u, s.capacity());
}

// kernel and functors for testing push_back functionality
namespace kernels {
    template <typename F>
    __global__
    void push_back(gpu::stack<int>& s, F f) {
        if (f(threadIdx.x)) {
            s.push_back(threadIdx.x);
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

    const unsigned n = 10;
    EXPECT_TRUE(n%2 == 0); // require n is even for tests to work
    auto s = gpu::make_managed_ptr<stack>(n);

    kernels::push_back<<<1, n>>>(*s, kernels::all_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n, s->size());
    for (auto i=0; i<int(s->size()); ++i) {
        EXPECT_EQ(i, (*s)[i]);
    }

    s->clear();
    kernels::push_back<<<1, n>>>(*s, kernels::even_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n/2, s->size());
    for (auto i=0; i<int(s->size())/2; ++i) {
        EXPECT_EQ(2*i, (*s)[i]);
    }

    s->clear();
    kernels::push_back<<<1, n>>>(*s, kernels::odd_ftor());
    cudaDeviceSynchronize();
    EXPECT_EQ(n/2, s->size());
    for (auto i=0; i<int(s->size())/2; ++i) {
        EXPECT_EQ(2*i+1, (*s)[i]);
    }
}
