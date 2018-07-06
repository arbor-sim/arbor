#include "../gtest.h"

#include <limits>
#include <type_traits>

#include <memory/memory.hpp>
#include <util/span.hpp>

//
//  wrappers
//

using namespace arb;

// test that memory::on_gpu copies std::vector into a device vector
TEST(vector, make_gpu_stdvector) {
    std::vector<int> stdvec(10);
    auto gpu_vec = memory::on_gpu(stdvec);
    using target_type = std::decay_t<decltype(gpu_vec)>;
    EXPECT_EQ(gpu_vec.size(), stdvec.size());
    EXPECT_NE(gpu_vec.data(), stdvec.data());
    EXPECT_TRUE(memory::util::is_on_gpu<target_type>());
    EXPECT_FALSE(memory::util::is_on_host<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_host copies a device vector into a host vector
TEST(vector, make_host_devicevector) {
    memory::device_vector<int> dvec(10);
    auto host_vec = memory::on_host(dvec);
    using target_type = std::decay_t<decltype(host_vec)>;
    EXPECT_EQ(host_vec.size(), dvec.size());
    EXPECT_NE(host_vec.data(), dvec.data());
    EXPECT_TRUE(memory::util::is_on_host<target_type>());
    EXPECT_FALSE(memory::util::is_on_gpu<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_gpu correctly makes a view of a device vector
/*
TEST(vector, make_gpu_devicevector) {
    memory::device_vector<int> dvec(10);
    auto view = memory::on_gpu(dvec);
    using target_type = std::decay_t<decltype(view)>;
    EXPECT_EQ(view.size(), dvec.size());
    EXPECT_EQ(view.data(), dvec.data());
    EXPECT_TRUE(memory::util::is_on_gpu<target_type>());
    EXPECT_FALSE(memory::util::is_on_host<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}
*/

//
//  fill
//

// test filling of memory with values on the gpu
TEST(vector, fill_gpu) {
    constexpr auto N = 10u;

    using util::make_span;
    // fill an array
    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::device_vector<double> v(n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(value, double(v[i]));
        }
    }

    // fill an array view
    /*
    memory::device_vector<float> ubervec(N);
    for (auto n : make_span(0u, N)) {
        float value = float((n+1)/2.f);
        // make a view of a sub-range of the std::vector ubervec
        auto v = ubervec(0, n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(float(v[i]), value);
        }
    }
    */
}

//
//  copy
//

TEST(vector, copy_h2d) {
    constexpr auto N = 10u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        std::vector<double> src(n, value);
        memory::device_vector<double> tgt(n);
        memory::fill(tgt, std::numeric_limits<double>::quiet_NaN());

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

TEST(vector, copy_d2h) {
    constexpr auto N = 20u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::device_vector<double> src(n);
        std::vector<double> tgt(n, std::numeric_limits<double>::quiet_NaN());
        memory::fill(src, value);

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

TEST(vector, copy_d2d) {
    constexpr auto N = 20u;

    using util::make_span;

    for (auto n : make_span(0u, N)) {
        double value = (n+1)/2.;
        memory::device_vector<double> src(n);
        memory::device_vector<double> tgt(n);
        memory::fill(src, value);
        memory::fill(tgt, std::numeric_limits<double>::quiet_NaN());

        memory::copy(src, tgt);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(double(tgt[i]), value);
        }
    }
}

