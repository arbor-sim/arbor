#include "gtest.h"

#include <DeviceCoordinator.hpp>
#include <HostCoordinator.hpp>

// verify that type members set correctly
TEST(DeviceCoordinator, type_members) {
    using namespace memory;

    typedef DeviceCoordinator<int> intcoord_t;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<int,   intcoord_t::value_type>();
}

// verify that rebinding works
TEST(DeviceCoordinator, rebind) {
    using namespace memory;

    using intcoord_t    = DeviceCoordinator<int>;
    using doublecoord_t = typename intcoord_t::rebind<double>;

    // verify that the correct type is used for internal storage
    ::testing::StaticAssertTypeEq<double,doublecoord_t::value_type>();
}

// test allocation of base arrays using host_coordinator
TEST(DeviceCoordinator, arraybase_alloc_free) {
    using namespace memory;

    typedef DeviceCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto array = coordinator.allocate(5);
    typedef decltype(array) arr_t;

    // test that array is a base array
    EXPECT_TRUE(impl::is_array_view<arr_t>::value);

    // test that array has correct storage type
    ::testing::StaticAssertTypeEq<int, arr_t::value_type >();

    // verify that the array has non-NULL pointer
    EXPECT_NE(arr_t::pointer(0), array.data())
        << "DeviceCoordinator returned a NULL pointer when allocating a nonzero array";

    // verify that freeing works
    coordinator.free(array);

    EXPECT_EQ(arr_t::pointer(0), array.data());
    EXPECT_EQ(arr_t::size_type(0), array.size());
}

// test allocation of reference arrays
TEST(DeviceCoordinator, refarray_alloc_free) {
    using namespace memory;

    typedef DeviceCoordinator<float> floatcoord_t;
    floatcoord_t coordinator;

    auto array = coordinator.allocate(5);
    typedef decltype(array) rng_t;

    auto ref_array = array(memory::all);
    typedef decltype(ref_array) rrng_t;

    // test that array has correct storage type
    ::testing::StaticAssertTypeEq<float, rrng_t::value_type >();

    // verify that the array has non-NULL pointer
    EXPECT_NE(rrng_t::pointer(0), ref_array.data())
        << "DeviceCoordinator returned a NULL pointer when allocating a nonzero array";

    EXPECT_EQ(array.data(), ref_array.data())
        << "base(all) does not have the same pointer address as base";

    // verify that freeing works
    coordinator.free(array);

    EXPECT_EQ(rng_t::pointer(0),   array.data());
    EXPECT_EQ(rng_t::size_type(0), array.size());
}

// test that DeviceCoordinator can correctly detect overlap between arrays
TEST(DeviceCoordinator, overlap) {
    using namespace memory;

    const int N = 20;

    typedef DeviceCoordinator<int> intcoord_t;
    intcoord_t coordinator;

    auto array = coordinator.allocate(N);
    auto array_other = coordinator.allocate(N);
    EXPECT_FALSE(array.overlaps(array_other));
    EXPECT_FALSE(array(0,10).overlaps(array(10,end)));
    EXPECT_FALSE(array(10,end).overlaps(array(0,10)));

    EXPECT_TRUE(array.overlaps(array));
    EXPECT_TRUE(array(memory::all).overlaps(array));
    EXPECT_TRUE(array.overlaps(array(memory::all)));
    EXPECT_TRUE(array(memory::all).overlaps(array(memory::all)));
    EXPECT_TRUE(array(0,11).overlaps(array(10,end)));
    EXPECT_TRUE(array(10,end).overlaps(array(0,11)));
}

// test copy from host to device memory works for unpinned host memory
TEST(DeviceCoordinator, host_to_device_copy_synchronous) {
    using namespace memory;

    const int N = 100;

    {
        typedef int T;
        typedef DeviceCoordinator<T> dc_t;
        typedef HostCoordinator<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        dc_t().copy(host_array, device_array);

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }

    {
        typedef double T;
        typedef DeviceCoordinator<T> dc_t;
        typedef HostCoordinator<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        auto event = dc_t().copy(host_array, device_array);
        //std::cout << util::type_printer<decltype(event)>::print() << std::endl;

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }
}

template <typename T>
using PinnedCoord = memory::HostCoordinator<T, memory::PinnedAllocator<T>>;

// test copy from host to device memory works for pinned host memory
TEST(DeviceCoordinator, host_to_device_copy_asynchronous) {
    using namespace memory;

    const int N = 100;

    {
        typedef int T;
        typedef DeviceCoordinator<T> dc_t;
        typedef PinnedCoord<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        auto event = dc_t().copy(host_array, device_array);

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }

    {
        typedef double T;
        typedef DeviceCoordinator<T> dc_t;
        typedef PinnedCoord<T>   hc_t;
        typedef ArrayView<T, dc_t> da_t;
        typedef ArrayView<T, hc_t> ha_t;

        // allocate array on host and device
        ha_t host_array(hc_t().allocate(N));
        da_t device_array(dc_t().allocate(N));

        // initialize host memory to linear sequence of integers
        for(auto i: Range(0,N))
            host_array[i] = T(i);

        // copy host array to device array
        auto event = dc_t().copy(host_array, device_array);

        //std::cout << util::type_printer<decltype(event)>::print() << std::endl;
        auto &event_ref = event.first;
        event_ref.wait();

        // check that host and device values are the same
        for(auto i: Range(0,N))
            EXPECT_EQ( host_array[i], T(device_array[i]) );
    }
}

// test copy from host to device memory works for pinned host memory
TEST(DeviceCoordinator, alignment) {
    using namespace memory;
    static_assert(DeviceCoordinator<double>::alignment() == 256,
                  "bad alignment reported by CUDA Allocator");
}

