#include "gtest.h"

#include <Array.hpp>
#include <HostCoordinator.hpp>

// verify that metafunctions for checking range wrappers work
TEST(Array, is_array) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_view;

    static_assert(
        impl::is_array_by_value<by_value>::value,
        "is_array_by_value incorrectly returned false for array_by_value" );

    static_assert(
        !impl::is_array_view<by_value>::value,
        "is_array_view incorrectly returned true for array_by_value" );

    static_assert(
        !impl::is_array_by_value<by_view>::value,
        "is_array_by_value incorrectly returned true for array_view" );

    static_assert(
        impl::is_array_view<by_view>::value,
        "is_array_view incorrectly returned false for array_view" );

    static_assert(
        impl::is_array<by_value>::value,
        "is_array incorrectly returned false for array_by_value" );

    static_assert(
        impl::is_array<by_view>::value,
        "is_array incorrectly returned false for array_view" );

    static_assert(
        !impl::is_array_by_value<int>::value,
        "is_array_by_value returns true for type other than array_by_value" );

    static_assert(
        !impl::is_array_view<int>::value,
        "is_array_view returns true for type other than array_by_value" );
}

TEST(Array,new_array_by_value) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_reference;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v1(10);
    EXPECT_EQ(v1.size(), 10);

    // create a copy
    by_value v2(v1);
    EXPECT_EQ(v2.size(), v1.size());
    EXPECT_NE(v2.data(), v1.data());

    by_reference r1(v1);
    EXPECT_EQ(r1.size(), v1.size());
    EXPECT_EQ(r1.data(), v1.data());
    by_value v3(r1);
    EXPECT_EQ(v3.size(), r1.size());
    EXPECT_NE(v3.data(), r1.data());
}

TEST(Array,new_array_by_ref) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_reference;

    // create range by value of length 10
    // this should allocate memory of length 10*sizeof(T)
    by_value v1(10);
    by_reference r1(v1);

    EXPECT_EQ(r1.size(), v1.size());
    EXPECT_EQ(r1.data(), v1.data());
    by_value v2(r1);
    EXPECT_EQ(v2.size(), v1.size());
    EXPECT_NE(v2.data(), v1.data());
}

TEST(Array, sub_ranges_by_index) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_reference;

    // check that reference range from a subrange works
    by_value v1(10);
    for(int i=0; i<10; i++)
        v1[i] = double(i);

    // create a reference wrapper to the second half of v1
    auto r1 = v1(5,end);
    EXPECT_EQ(r1.size(), 5);
    EXPECT_EQ(v1.data()+5, r1.data());

    // create a value wrapper (copy) of the reference range
    // should contain a copy of the second half of v1
    by_value v2(r1);
    EXPECT_NE(r1.data(), v2.data());
    EXPECT_EQ(r1.size(), v2.size());
    auto it1 = v1.data();
    auto it2 = v2.data();
    for(auto i=0; i<5; ++i) {
        EXPECT_EQ(it1[i+5], it2[i]);
    }
}

// test use of Range type to get views on an Array
TEST(Array, sub_ranges_by_range) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_reference;

    // check that a view of a subrange works
    by_value array1(10);
    for(int i=0; i<10; i++)
        array1[i] = double(i);

    Range range1(5,10);
    auto view1 = array1(range1);
    EXPECT_EQ(view1.size(), 5);
    EXPECT_EQ(array1.data()+5, view1.data());

    // create a copy of the data in a view
    // should contain a copy of the second half of array1
    by_value array2(view1);
    EXPECT_NE(view1.data(), array2.data());
    EXPECT_EQ(view1.size(), array2.size());
    auto it1 = array1.data()+5;
    auto it2 = array2.data()+5;
    while(it2!=array2.data()+array2.size()) {
        EXPECT_EQ(*it1, *it2);
        ++it2;
        ++it1;
    }
}

// test that the assignment operator works
TEST(Array, assignment) {
    using namespace memory;

    typedef Array<double, HostCoordinator<double> > by_value;
    typedef ArrayView<double, HostCoordinator<double> > by_reference;

    by_value other(10);
    other(all) = 3.14;
    by_value v;

    v = other(all);
    EXPECT_NE(v.data(), other.data());
    for(auto value: v)
        EXPECT_EQ(value, 3.14);
}
