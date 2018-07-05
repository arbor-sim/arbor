#include "../gtest.h"

#include <limits>
#include <type_traits>

#include <memory/memory.hpp>
#include <util/span.hpp>

//
//  wrappers
//

using namespace arb;

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
    using target_type = std::decay_t<decltype(host_vec)>;
    EXPECT_EQ(host_vec.size(), stdvec.size());
    EXPECT_EQ(host_vec.data(), stdvec.data());
    EXPECT_TRUE(memory::util::is_on_host<target_type>());

    EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
}

// test that memory::on_host makes a view of host_vector and host view
TEST(vector, make_host_hostvector) {
    memory::host_vector<int> vec(10);
    {   // test from host_vector
        auto host_view = memory::on_host(vec);
        using target_type = std::decay_t<decltype(host_view)>;
        EXPECT_EQ(host_view.size(), vec.size());
        EXPECT_EQ(host_view.data(), vec.data());
        EXPECT_TRUE(memory::util::is_on_host<target_type>());

        EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
    }
    {   // test from view
        auto view = memory::make_view(vec);
        auto host_view = memory::on_host(view);
        using target_type = std::decay_t<decltype(host_view)>;
        EXPECT_EQ(host_view.size(), view.size());
        EXPECT_EQ(host_view.data(), view.data());
        EXPECT_TRUE(memory::util::is_on_host<target_type>());

        EXPECT_TRUE((std::is_same<int, target_type::value_type>::value));
    }
}

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
        memory::host_vector<double> v(n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(v[i], value);
        }
    }

    // fill an array view
    std::vector<float> ubervec(N);
    for (auto n : make_span(0u, N)) {
        float value = float((n+1)/2.f);
        using view_type = memory::host_vector<float>::view_type;
        // make a view of a sub-range of the std::vector ubervec
        auto v = view_type(ubervec.data(), n);
        memory::fill(v, value);

        for (auto i: make_span(0u, n)) {
            EXPECT_EQ(v[i], value);
        }
    }
}

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

