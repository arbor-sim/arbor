#include <cstdint>

#include <util/padded_alloc.hpp>

#include "../gtest.h"
#include "common.hpp"
#include "instrument_malloc.hpp"

using arb::util::padded_allocator;

template <typename T>
using pvector = std::vector<T, padded_allocator<T>>;

// (For k a power of 2 only)
static bool is_aligned(void* p, std::size_t k) {
    auto addr = reinterpret_cast<std::uintptr_t>(p);
    return !(addr&(k-1));
}

TEST(padded_vector, alignment) {
    padded_allocator<double> pa(1024);
    pvector<double> a(101, 0.0, pa);

    EXPECT_EQ(1024u, a.get_allocator().alignment());
    EXPECT_TRUE(is_aligned(a.data(), 1024));
}

TEST(padded_vector, allocator_constraints) {
    EXPECT_THROW(padded_allocator<char>(7), std::range_error);

    padded_allocator<char> pa(2); // less than sizeof(void*)
    std::vector<char, padded_allocator<char>> v(7, 'a', pa);

    EXPECT_TRUE(is_aligned(v.data(), sizeof(void*)));
}

TEST(padded_vector, allocator_propagation) {
    padded_allocator<double> pa(1024);
    pvector<double> a(101, 0, pa);

    EXPECT_EQ(pa, a.get_allocator());

    pvector<double> b(101);
    auto pb = b.get_allocator();

    // Differing alignment => allocators compare not-equal.
    EXPECT_EQ(1u, pb.alignment());
    EXPECT_NE(pa, pb);

    // Propagate on copy- or move-assignment:
    b = a;
    EXPECT_NE(pb.alignment(), b.get_allocator().alignment());
    EXPECT_EQ(pa.alignment(), b.get_allocator().alignment());

    pvector<double> c;
    c = std::move(a);
    EXPECT_EQ(c.get_allocator().alignment(), pa.alignment());
}


#ifdef CAN_INSTRUMENT_MALLOC

struct alloc_data {
    unsigned n_malloc = 0;
    unsigned n_realloc = 0;
    unsigned n_memalign = 0;
    unsigned n_free = 0;

    std::size_t last_malloc = -1;
    std::size_t last_realloc = -1;
    std::size_t last_memalign = -1;
};

struct count_allocs: testing::with_instrumented_malloc {
    alloc_data data;

    void on_malloc(std::size_t size, const void*) override {
        ++data.n_malloc;
        data.last_malloc = size;
    }

    void on_realloc(void*, std::size_t size, const void*) override {
        ++data.n_realloc;
        data.last_realloc = size;
    }

    void on_memalign(std::size_t, std::size_t size, const void*) override {
        ++data.n_memalign;
        data.last_memalign = size;
    }

    void on_free(void*, const void*) override {
        ++data.n_free;
    }

    void reset() {
        data = alloc_data();
    }
};

TEST(padded_vector, instrumented) {
    count_allocs A;

    padded_allocator<double> pad256(256), pad32(32);
    pvector<double> v1p256(303, pad256);
    alloc_data mdata = A.data;

    unsigned expected_v1_alloc = 303*sizeof(double);
    expected_v1_alloc = expected_v1_alloc%256? 256*(1+expected_v1_alloc/256): expected_v1_alloc;

    EXPECT_EQ(1u, mdata.n_memalign);
    EXPECT_EQ(0u, mdata.n_malloc);
    EXPECT_EQ(0u, mdata.n_realloc);
    EXPECT_EQ(expected_v1_alloc, mdata.last_memalign);

    // Move assignment: allocators propagate, so we do not expect v2
    // to perform a new allocation.

    pvector<double> v2p32(10, pad32);
    A.reset();
    v2p32 = std::move(v1p256);
    mdata = A.data;

    EXPECT_EQ(0u, mdata.n_memalign);
    EXPECT_EQ(0u, mdata.n_malloc);
    EXPECT_EQ(0u, mdata.n_realloc);

    pvector<double> v3p256(101, pad256), v4p256(700, pad256);

    A.reset();
    v4p256 = v3p256; // same alignment, larger size => shouldn't need to allocate
    mdata = A.data;

    EXPECT_EQ(0u, mdata.n_memalign);
    EXPECT_EQ(0u, mdata.n_malloc);
    EXPECT_EQ(0u, mdata.n_realloc);

    A.reset();
    pvector<double> v5p32(701, pad32);
    mdata = A.data;

    unsigned expected_v5_alloc = 701*sizeof(double);
    expected_v5_alloc = expected_v5_alloc%32? 32*(1+expected_v5_alloc/32): expected_v5_alloc;

    EXPECT_EQ(1u, mdata.n_memalign);
    EXPECT_EQ(0u, mdata.n_malloc);
    EXPECT_EQ(0u, mdata.n_realloc);
    EXPECT_EQ(expected_v5_alloc, mdata.last_memalign);

    A.reset();
    v5p32 = v3p256; // enough space, but different alignment, so should free and then allocate.
    mdata = A.data;

    EXPECT_EQ(1u, mdata.n_free);
    EXPECT_EQ(1u, mdata.n_memalign);
    EXPECT_EQ(0u, mdata.n_malloc);
    EXPECT_EQ(0u, mdata.n_realloc);
}

#endif // ifdef CAN_INSTRUMENT_MALLOC
