#include "gtest.h"

#include <type_traits>

#include <vector/Vector.hpp>
#include <vector/helpers.hpp>

//
//  helpers
//

// test that memory::make_view and make_const_view work on std::vector
TEST(vector, make_view_stdvector) {
    // test that we can make views of std::vector
    std::vector<int> stdvec(10);
    auto view = memory::make_view(stdvec);
    EXPECT_EQ(view.size(), stdvec.size());
    EXPECT_EQ(view.data(), stdvec.data());
    EXPECT_TRUE((std::is_same<int*, decltype(view.data())>::value));

    auto const_view = memory::make_const_view(stdvec);
    EXPECT_EQ(const_view.size(), stdvec.size());
    EXPECT_EQ(const_view.data(), stdvec.data());
    EXPECT_TRUE((std::is_same<const int*, decltype(const_view.data())>::value));
}

// test that memory::on_host makes a view of std::vector
TEST(vector, make_host_stdvector) {
    std::vector<int> stdvec(10);
    auto host_vec = memory::on_host(stdvec);
    using target_type = std::decay<decltype(host_vec)>::type;
    EXPECT_EQ(host_vec.size(), stdvec.size());
    EXPECT_EQ(host_vec.data(), stdvec.data());
    EXPECT_TRUE(memory::util::is_on_host<target_type>());
    EXPECT_FALSE(memory::util::is_on_gpu<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_host makes a view of HostVector and host view
TEST(vector, make_host_hostvector) {
    memory::HostVector<int> vec(10);
    {   // test from HostVector
        auto host_view = memory::on_host(vec);
        using target_type = std::decay<decltype(host_view)>::type;
        EXPECT_EQ(host_view.size(), vec.size());
        EXPECT_EQ(host_view.data(), vec.data());
        EXPECT_TRUE(memory::util::is_on_host<target_type>());
        EXPECT_FALSE(memory::util::is_on_gpu<target_type>());

        EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
    }
    {   // test from view
        auto view = memory::make_view(vec);
        auto host_view = memory::on_host(view);
        using target_type = std::decay<decltype(host_view)>::type;
        EXPECT_EQ(host_view.size(), view.size());
        EXPECT_EQ(host_view.data(), view.data());
        EXPECT_TRUE(memory::util::is_on_host<target_type>());
        EXPECT_FALSE(memory::util::is_on_gpu<target_type>());

        EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
    }
}

#ifdef WITH_CUDA
// test that memory::on_gpu copies std::vector into a device vector
TEST(vector, make_gpu_stdvector) {
    std::vector<int> stdvec(10);
    auto gpu_vec = memory::on_gpu(stdvec);
    using target_type = std::decay<decltype(gpu_vec)>::type;
    EXPECT_EQ(gpu_vec.size(), stdvec.size());
    EXPECT_NE(gpu_vec.data(), stdvec.data());
    EXPECT_TRUE(memory::util::is_on_gpu<target_type>());
    EXPECT_FALSE(memory::util::is_on_host<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_host copies a device vector into a host vector
TEST(vector, make_host_devicevector) {
    memory::DeviceVector<int> dvec(10);
    auto host_vec = memory::on_host(dvec);
    using target_type = std::decay<decltype(host_vec)>::type;
    EXPECT_EQ(host_vec.size(), dvec.size());
    EXPECT_NE(host_vec.data(), dvec.data());
    EXPECT_TRUE(memory::util::is_on_host<target_type>());
    EXPECT_FALSE(memory::util::is_on_gpu<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_gpu correctly makes a view of a device vector
TEST(vector, make_gpu_devicevector) {
    memory::DeviceVector<int> dvec(10);
    auto view = memory::on_gpu(dvec);
    using target_type = std::decay<decltype(view)>::type;
    EXPECT_EQ(view.size(), dvec.size());
    EXPECT_EQ(view.data(), dvec.data());
    EXPECT_TRUE(memory::util::is_on_gpu<target_type>());
    EXPECT_FALSE(memory::util::is_on_host<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}
#endif
