#include <gtest/gtest.h>

#include "backends/gpu/stack.hpp"
#include "backends/gpu/stack_cu.hpp"
#include "gpu_context.hpp"

using namespace arb;

TEST(stack, construction) {
    using T = int;

    auto context = make_gpu_context(0);
    if (!context->has_gpu()) return;

    gpu::stack<T> s(10, context);

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

    auto context = make_gpu_context(0);
    if (!context->has_gpu()) return;

    const unsigned n = 10;
    EXPECT_TRUE(n%2 == 0); // require n is even for tests to work
    auto s = stack(n, context);
    auto& sstorage = s.storage();

    EXPECT_EQ(0u, s.size()); // dummy tests
    EXPECT_EQ(n, s.capacity());


    kernels::push_back<<<1, n>>>(sstorage, kernels::all_ftor());
    s.update_host();
    EXPECT_EQ(n, s.size());
    {
        auto d = s.data();
        EXPECT_EQ(s.size(), d.size());
        std::sort(d.begin(), d.end());
        for (unsigned i=0; i<n; ++i) {
            EXPECT_EQ(i, d[i]);
        }
    }

    s.clear();
    kernels::push_back<<<1, n>>>(sstorage, kernels::even_ftor());
    s.update_host();
    EXPECT_EQ(n/2, s.size());
    {
        auto d = s.data();
        EXPECT_EQ(s.size(), d.size());
        std::sort(d.begin(), d.end());
        for (unsigned i=0; i<n/2; ++i) {
            EXPECT_EQ(2*i, d[i]);
        }
    }

    s.clear();
    kernels::push_back<<<1, n>>>(sstorage, kernels::odd_ftor());
    s.update_host();
    EXPECT_EQ(n/2, s.size());
    {
        auto d = s.data();
        EXPECT_EQ(s.size(), d.size());
        std::sort(d.begin(), d.end());
        for (unsigned i=0; i<n/2; ++i) {
            EXPECT_EQ(2*i+1, d[i]);
        }
    }
}

TEST(stack, overflow) {
    using T = int;
    using stack = gpu::stack<T>;

    auto context = make_gpu_context(0);
    if (!context->has_gpu()) return;

    const unsigned n = 10;
    auto s = stack(n, context);
    auto& sstorage = s.storage();
    EXPECT_FALSE(s.overflow());

    // push 2n items into a stack of size n
    kernels::push_back<<<1, 2*n>>>(sstorage, kernels::all_ftor());
    s.update_host();
    EXPECT_EQ(n, s.size());
    EXPECT_EQ(2*n, s.pushes());
    EXPECT_TRUE(s.overflow());
}

TEST(stack, empty) {
    using T = int;
    using stack = gpu::stack<T>;

    auto context = make_gpu_context(0);
    if (!context->has_gpu()) return;

    stack s(0u, context);

    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.capacity(), 0u);

    auto device_storage = s.get_storage_copy();

    EXPECT_EQ(device_storage.data, nullptr);
}
