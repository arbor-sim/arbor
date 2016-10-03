#include "gtest.h"

#include <HostCoordinator.hpp>

// helper function for outputting a range
template <typename R>
void print_range(const R& rng) {
    for(auto v: rng)
        std::cout << v << " ";
    std::cout << std::endl;
}

// verify that type members set correctly
TEST(HostCoordinator, type_members) {
    using namespace memory;

    typedef HostCoordinator<int> intcoord_t;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<int,   intcoord_t::value_type>();
}

// verify that rebinding works
TEST(HostCoordinator, rebind) {
    using namespace memory;

    typedef HostCoordinator<int> intcoord_t;
    typedef typename intcoord_t::rebind<double> doublecoord_t;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<double,doublecoord_t::value_type>();
}

// test allocation of base ranges using HostCoordinator
TEST(HostCoordinator, baserange_alloc_free) {
    using namespace memory;

    typedef HostCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto rng = coordinator.allocate(5);
    typedef decltype(rng) rng_t;

    intcoord_t coord;

    // test that range is a base range
    EXPECT_TRUE(impl::is_array_view<rng_t>::value);

    // test that range has correct storage type
    ::testing::StaticAssertTypeEq<int, rng_t::value_type >();

    // verify that the range has non-NULL pointer
    EXPECT_NE(rng_t::pointer(0), rng.data()) << "HostCoordinator returned a NULL pointer when alloating a nonzero range";

    // verify that freeing works
    coordinator.free(rng);

    EXPECT_EQ(rng_t::pointer(0), rng.data());
    EXPECT_EQ(rng_t::size_type(0), rng.size());
}

// test allocation of reference ranges
TEST(HostCoordinator, refrange_alloc_free) {
    using namespace memory;

    typedef HostCoordinator<float> floatcoord_t;
    floatcoord_t coordinator;

    auto rng = coordinator.allocate(5);
    typedef decltype(rng) rng_t;

    auto rrng = rng(all);
    typedef decltype(rrng) rrng_t;

    // test that range has correct storage type
    ::testing::StaticAssertTypeEq<float, rrng_t::value_type >();

    // verify that the range has non-NULL pointer
    EXPECT_NE(rrng_t::pointer(0), rrng.data())
        << "HostCoordinator returned a NULL pointer when allocating a nonzero range";

    EXPECT_EQ(rng.data(), rrng.data())
        << "base(all) does not have the same pointer address as base";

    // verify that freeing works
    coordinator.free(rng);

    EXPECT_EQ(rng_t::pointer(0),   rng.data());
    EXPECT_EQ(rng_t::size_type(0), rng.size());
}

// test that HostCoordinator can correctly detect overlap between ranges
TEST(HostCoordinator, overlap) {
    using namespace memory;

    const int N = 20;

    typedef HostCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto rng = coordinator.allocate(N);
    auto rng_other = coordinator.allocate(N);
    EXPECT_FALSE(rng.overlaps(rng_other));
    EXPECT_FALSE(rng(0,10).overlaps(rng(10,end)));
    EXPECT_FALSE(rng(10,end).overlaps(rng(0,10)));

    EXPECT_TRUE(rng.overlaps(rng));
    EXPECT_TRUE(rng(all).overlaps(rng));
    EXPECT_TRUE(rng.overlaps(rng(all)));
    EXPECT_TRUE(rng(all).overlaps(rng(all)));
    EXPECT_TRUE(rng(0,11).overlaps(rng(10,end)));
    EXPECT_TRUE(rng(10,end).overlaps(rng(0,11)));
}

// test copy from host to device memory works for pinned host memory
TEST(HostCoordinator, alignment) {
    using namespace memory;
    static_assert(
      HostCoordinator<double, AlignedAllocator<double,64>>::alignment() == 64,
      "bad alignment reported by Host Allocator");
    static_assert(
      HostCoordinator<double, AlignedAllocator<double,128>>::alignment() == 128,
      "bad alignment reported by Host Allocator");
    static_assert(
      HostCoordinator<double, AlignedAllocator<double,256>>::alignment() == 256,
      "bad alignment reported by Host Allocator");
    static_assert(
      HostCoordinator<double, AlignedAllocator<double,512>>::alignment() == 512,
      "bad alignment reported by Host Allocator");
}
