#include <cstdint>

#if (__GLIBC__==2)
#include <malloc.h>
#define INSTRUMENT_MALLOC
#endif

#include <util/padded_alloc.hpp>

#include "../gtest.h"
#include "common.hpp"
#include "instrument_malloc.hpp"

using arb::util::padded_allocator;

template <typename T>
using pvector = std::vector<T, padded_allocator<T>>;

static bool is_aligned(void* p, std::size_t k) {
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    return !(addr&(k-1));
}

TEST(padded_vector, alignment) {
    padded_allocator<double> pa(1024);
    pvector<double> a(101, pa);

    EXPECT_EQ(1024u, a.get_allocator().alignment());
    EXPECT_TRUE(is_aligned(a.data(), 1024));
}

TEST(padded_vector, allocator_propagation) {
    padded_allocator<double> pa(1024);
    pvector<double> a(101, pa);

    EXPECT_EQ(pa, a.get_allocator());

    pvector<double> b(101);
    auto pb = b.get_allocator();

    EXPECT_EQ(1u, pb.alignment());
    EXPECT_NE(pa, pb);

    b = a;
    EXPECT_EQ(1024u, b.get_allocator().alignment());
    EXPECT_TRUE(is_aligned(b.data(), 1024));
    EXPECT_EQ(pa, b.get_allocator());
    EXPECT_NE(pb, b.get_allocator());

    pvector<double> c;
    c = std::move(a);

    EXPECT_EQ(pa, c.get_allocator());
}


#ifdef INSTRUMENT_MALLOC

struct count_allocs: testing::with_instrumented_malloc {
    unsigned n_malloc = 0;
    unsigned n_realloc = 0;
    unsigned n_memalign = 0;

    std::size_t last_malloc = -1;
    std::size_t last_realloc = -1;
    std::size_t last_memalign = -1;

    void on_malloc(std::size_t size, const void*) override {
        ++n_malloc;
        last_malloc = size;
    }

    void on_realloc(void*, std::size_t size, const void*) override {
        ++n_realloc;
        last_realloc = size;
    }

    void on_memalign(std::size_t, std::size_t size, const void*) override {
        ++n_memalign;
        last_memalign = size;
    }
};

TEST(padded_vector, instrumented) {
    count_allocs A;

    padded_allocator<double> pad256(256), pad32(32);
    pvector<double> v1(303, pad256);

    unsigned expected_v1_alloc = 303*sizeof(double);
    expected_v1_alloc = expected_v1_alloc%256? 256*(1+expected_v1_alloc/256): expected_v1_alloc;

    EXPECT_EQ(1u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);
    EXPECT_EQ(expected_v1_alloc, A.last_memalign);

    pvector<double> v2(pad32);
    v2 = std::move(v1); // move should move allocator and not copy

    EXPECT_EQ(1u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);

    pvector<double> v3(700, pad256);

    EXPECT_EQ(2u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);

    v3 = v2; // same alignment, larger size => shouldn't need to allocate

    EXPECT_EQ(2u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);

    pvector<double> v4(700, pad32);

    EXPECT_EQ(3u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);
    EXPECT_NE(expected_v1_alloc, A.last_memalign);

    v4 = v2; // different alignment, so will have to reallocate

    EXPECT_EQ(4u, A.n_memalign);
    EXPECT_EQ(0u, A.n_malloc);
    EXPECT_EQ(0u, A.n_realloc);
    EXPECT_EQ(expected_v1_alloc, A.last_memalign);
}

#endif // ifdef INSTRUMENT_MALLOC
