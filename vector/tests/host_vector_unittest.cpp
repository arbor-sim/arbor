#include "gtest.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <Vector.hpp>
#include <HostCoordinator.hpp>

template <typename VEC>
void print(VEC const& v) {
    for(auto v_: v)
        std::cout << v_ << " ";
    std::cout << std::endl;
}

// test that constructors work
TEST(HostVector, constructor) {
    using namespace memory;

    // default constructor
    HostVector<float> v2;

    // length constructor
    HostVector<float> v1(100);

    // initialize values as monotone sequence
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    // initialize new HostVector from a subrange
    HostVector<float> v3(v1(90, 100));

    // reset values in range
    for(auto &v : v1(90, 100))
        v = float(-1.0);

    // check that v3 has values originally copied over from v1
    for(int i=0; i<10; i++)
        EXPECT_EQ(float(i+90), v3[i]);

    for(int i=90; i<100; i++)
        EXPECT_EQ(float(-1), v1[i]);
}

// test that constructors work
TEST(HostVector, std_vector_constructor) {
    using namespace memory;

    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);

    HostVector<int> hv(v);

    for(auto i=0; i<hv.size(); ++i) {
        EXPECT_EQ(v[i], hv[i]);
    }
}

// test that copy constructors work
TEST(HostVector, copy_constructor) {
    using namespace memory;

    // length constructor
    HostVector<float> v1(100);

    // initialize values as monotone sequence
    for(int i=0; i<v1.size(); ++i)
        v1[i] = float(i);

    // copy constructor
    //HostVector<float> v2;
    //v2 = v1;
    HostVector<float> v2 = v1;

    // ensure that new memory was allocated
    EXPECT_NE(v2.data(), v1.data());

    // check that v3 has values originally copied over from v1
    for(int i=0; i<100; i++)
        EXPECT_EQ(v1[i], v2[i]);
}

template <typename T, size_t N>
using vecN =
    memory::Array<T, memory::HostCoordinator<T, memory::AlignedAllocator<T, N>>>;

// test that copy works between vectors with different alignments
TEST(HostVector, copy) {
    using namespace memory;

    vecN<float, 16> v16(100);
    v16(memory::all) = 3.14f;
    vecN<float, 32> v32 = v16;

    for(auto v: v32) {
        EXPECT_EQ(v, 3.14f);
    }
}

// test that move constructors work
TEST(HostVector, move_constructor) {
    using namespace memory;

    // move constructor
    HostVector<float> v1 = HostVector<float>(100);

    EXPECT_EQ(v1.size(), 100);
}

// test that constructors with default value works
TEST(HostVector, value_constructor) {
    using namespace memory;

    HostVector<int> v1(10, -5);

    EXPECT_EQ(v1.size(), 10);
    for(auto i : v1.range()) {
        EXPECT_EQ(v1[i], -5);
    }
}

// test that iterators and ranges work
TEST(HostVector, iterators_and_ranges) {
    using namespace memory;

    // length constructor
    HostVector<float> v1(100);

    // check that begin()/end() iterators work
    for(auto it=v1.begin(); it<v1.end(); ++it)
        *it = float(3.0);

    // check that range based for loop works
    for(auto &val : v1)
        val = float(3.0);
    {
        float sum = 0;
        // check it works for const
        for(auto val : v1)
            sum+=val;
        EXPECT_EQ(float(3*100), sum);
    }

    // check that std::for_each works
    std::for_each(v1.begin(), v1.end(), [] (float& val) {val+=1;}); // add 1 to every value in v1
    {
        float sum = 0;
        for(auto val : v1)
            sum+=val;
        EXPECT_EQ(float(4*100), sum);
    }
}

TEST(HostVector, from_std_vector) {
    using namespace memory;

    /*
    const std::vector<int> svec{0,1,2,3,4,5,6,7};
    const HostVector<int>::view_type v(const_cast<int*>(svec.data()), 5);
    */

    std::vector<int> v{0,1,2,3,4,5,6,7};
    HostVector<int> hv(v);

    EXPECT_EQ(v.size(), hv.size());
    for(auto i : hv.range()) {
        EXPECT_EQ(v[i], hv[i]);
    }
}
