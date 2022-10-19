#include <cstdint>

#include <util/padded_alloc.hpp>

#include <gtest/gtest.h>
#include "common.hpp"

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
