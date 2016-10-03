#include "gtest.h"

#include <Vector.hpp>
#include <HostCoordinator.hpp>
#include <DeviceCoordinator.hpp>

#include <algorithm>

// test that constructors work
TEST(DeviceVector, constructor) {
    using namespace memory;

    // default constructor
    DeviceVector<float> v0;

    // length constructor
    const size_t N = 10;
    DeviceVector<float> v1(N);
    EXPECT_EQ(N, v1.size());

    // copy constructor
    DeviceVector<float> v2(v1);
    EXPECT_EQ(N, v2.size());

    // range constructor
    Range r(0,N/2);
    DeviceVector<float> v3(v1(r));
    EXPECT_EQ(r.size(), v3.size());
}

TEST(DeviceVector, indexing) {
    using namespace memory;

    // length constructor
    const size_t N = 10;
    DeviceVector<float> v1(N);

    for(int i=0; i<N; ++i)
        v1[i] = i;

    for(int i=0; i<N; ++i)
        EXPECT_EQ(float(v1[i]), (float)i);
}

TEST(DeviceVector, fill) {
    using namespace memory;

    // length constructor
    const size_t N = 10;

    {
        DeviceVector<char> v(N);
        v(memory::all) = 'a';
        for(int i=0; i<N; ++i)
            EXPECT_EQ(char(v[i]), 'a');
    }
    {
        DeviceVector<float> v(N);
        v(memory::all) = -1.f;
        for(int i=0; i<N; ++i)
            EXPECT_EQ(float(v[i]), -1.f);
    }
    {
        DeviceVector<double> v(N);
        v(memory::all) = -2.;
        for(int i=0; i<N; ++i)
            EXPECT_EQ(double(v[i]), -2.);
    }
}
