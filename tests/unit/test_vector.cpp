#include "gtest.h"

#include <limits>
#include <type_traits>

#include <memory/memory.hpp>
#include <util/span.hpp>

//
//  wrappers
//

using namespace nest::mc;

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

//
//  fill
//

// test filling of memory with values on the host
TEST(vector, fill_host) {
    constexpr auto N = 10u;

    using util::make_span;
    // fill a std::vector
    for (auto n : make_span(0u, N)) {
        std::vector<int> v(n, 0);
        memory::fill(v, 42);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(v[i], 42);
        }
    }

    // fill an array
    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::HostVector<double> v(n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(v[i], value);
        }
    }

    // fill an array view
    std::vector<float> ubervec(N);
    for (auto n : make_span(0u, N)) {
        float value = float((n+1)/2.f);
        using view_type = memory::HostVector<float>::view_type;
        // make a view of a sub-range of the std::vector ubervec
        auto v = view_type(ubervec.data(), n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(v[i], value);
        }
    }
}

#ifdef WITH_CUDA
// test filling of memory with values on the gpu
TEST(vector, fill_gpu) {
    constexpr auto N = 10u;

    using util::make_span;
    // fill an array
    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::DeviceVector<double> v(n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(v[i]), value);
        }
    }

    // fill an array view
    memory::DeviceVector<float> ubervec(N);
    for (auto n : make_span(0u, N)) {
        float value = float((n+1)/2.f);
        // make a view of a sub-range of the std::vector ubervec
        auto v = ubervec(0, n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(float(v[i]), value);
        }
    }
}
#endif

//
//  copy
//

TEST(vector, copy_h2h) {
    constexpr auto N = 10u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        std::vector<double> src(n, value);
        std::vector<double> tgt(n, std::numeric_limits<double>::quiet_NaN());

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(tgt[i], value);
        }
    }
}

TEST(vector, copy_h2d) {
    constexpr auto N = 10u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        std::vector<double> src(n, value);
        memory::DeviceVector<double> tgt(n);
        memory::fill(tgt, std::numeric_limits<double>::quiet_NaN());

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

TEST(vector, copy_d2h) {
    constexpr auto N = 10u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::DeviceVector<double> src(n);
        std::vector<double> tgt(n, std::numeric_limits<double>::quiet_NaN());
        memory::fill(src, value);

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

TEST(vector, copy_d2d) {
    constexpr auto N = 10u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::DeviceVector<double> src(n);
        memory::DeviceVector<double> tgt(n);
        memory::fill(src, value);
        memory::fill(tgt, std::numeric_limits<double>::quiet_NaN());

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

