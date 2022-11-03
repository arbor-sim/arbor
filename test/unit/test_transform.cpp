#include <gtest/gtest.h>

#include <cctype>
#include <forward_list>
#include <vector>

#include <util/range.hpp>
#include <util/indirect.hpp>
#include <util/span.hpp>
#include <util/transform.hpp>

#include "common.hpp"

using namespace arb;

TEST(transform, transform_view) {
    std::forward_list<int> fl = {1, 4, 6, 8, 10 };
    std::vector<double> result;

    auto r = util::transform_view(fl, [](int i) { return i*i+0.5; });

    EXPECT_EQ(5u, std::size(r));
    EXPECT_EQ(16.5, *(std::next(std::begin(r), 1)));

    std::copy(r.begin(), r.end(), std::back_inserter(result));
    std::vector<double> expected = { 1.5, 16.5, 36.5, 64.5, 100.5 };

    EXPECT_EQ(expected, result);
}

struct null_terminated_t {
    bool operator==(const char *p) const { return !*p; }
    bool operator!=(const char *p) const { return !!*p; }

    friend bool operator==(const char *p, null_terminated_t x) {
        return x==p;
    }

    friend bool operator!=(const char *p, null_terminated_t x) {
        return x!=p;
    }

    constexpr null_terminated_t() {}
};

constexpr null_terminated_t null_terminated;

char upper(char c) { return std::toupper(c); }

TEST(transform, transform_view_sentinel) {
    const char *hello = "hello";
    auto r = util::transform_view(util::make_range(hello, null_terminated), upper);

    std::string out;
    for (auto i = r.begin(); i!=r.end(); ++i) {
        out += *i;
    }
    EXPECT_EQ("HELLO", out);
}

TEST(transform, pointer_access) {
    struct item {
        int x;
    };
    item data[3];

    auto r = util::transform_view(util::make_span(0u, 3u), [&](unsigned i) -> item& { return data[i]; });

    int c=10;
    for (auto i=r.begin(); i!=r.end(); ++i) {
        i->x = c++;
    }

    EXPECT_EQ(10, data[0].x);
    EXPECT_EQ(11, data[1].x);
    EXPECT_EQ(12, data[2].x);
}

TEST(transform, pointer_proxy) {
    struct item {
        int x;
    };
    auto r = util::transform_view(util::make_span(0, 3), [&](int i) { return item{13+i}; });

    int c=13;
    for (auto i=r.begin(); i!=r.end(); ++i) {
        EXPECT_EQ(c++, i->x);
    }
}

TEST(indirect, fwd_index) {
    std::istringstream string_indices("5 2 3 0 1 1 4");
    const double data[6] = {10., 11., 12., 13., 14., 15.};

    auto indices = util::make_range(std::istream_iterator<int>(string_indices), std::istream_iterator<int>());
    auto permuted = util::indirect_view(data, indices);

    std::vector<double> result(permuted.begin(), permuted.end());
    std::vector<double> expected = {15., 12., 13., 10., 11., 11., 14.};

    EXPECT_EQ(expected, result);
}

TEST(indirect, nocopy) {
    const testing::nocopy<double> data[6] = {10., 11., 12., 13., 14., 15.};
    unsigned map_reverse[6] = {5, 4, 3, 2, 1, 0};
    auto reversed = util::indirect_view(data, map_reverse);

    std::vector<double> expected = {15., 14., 13., 12., 11., 10.};
    std::vector<double> result;
    for (auto& elem: reversed) { result.push_back(elem.value); }
    EXPECT_EQ(expected, result);

    unsigned map_evens[3] = {0, 2, 4};
    auto even_reversed = util::indirect_view(reversed, map_evens);

    expected = {15., 13., 11.};
    result.clear();
    for (auto& elem: even_reversed) { result.push_back(elem.value); }
    EXPECT_EQ(expected, result);
}

TEST(indirect, nomove) {
    testing::nomove<double> data[6];
    for (unsigned i=0; i<std::size(data); ++i) data[i].value = 10.+i;
    unsigned map_reverse[6] = {5, 4, 3, 2, 1, 0};
    auto reversed = util::indirect_view(data, map_reverse);

    std::vector<double> expected = {15., 14., 13., 12., 11., 10.};
    std::vector<double> result;
    for (auto& elem: reversed) { result.push_back(elem.value); }
    EXPECT_EQ(expected, result);

    unsigned map_evens[3] = {0, 2, 4};
    auto even_reversed = util::indirect_view(reversed, map_evens);

    expected = {15., 13., 11.};
    result.clear();
    for (auto& elem: even_reversed) { result.push_back(elem.value); }
    EXPECT_EQ(expected, result);
}

TEST(indirect, modifying) {
    unsigned map1[] = {0, 2, 4, 1, 3, 0};
    unsigned map2[] = {0, 1, 1, 1, 2};

    std::vector<double> data = {-1, -1, -1};

    auto permuted = util::indirect_view(util::indirect_view(data, map2), map1);

    // expected mapping:
    // permuted[0] = data[0]
    // permuted[1] = data[1]
    // permuted[2] = data[2]
    // permuted[3] = data[1]
    // permuted[4] = data[1]
    // permuted[5] = data[0]

    for (unsigned i = 0; i<std::size(permuted); ++i) {
        permuted[i] = 10.+i;
    }
    std::vector<double> expected = {15., 14., 12.};

    EXPECT_EQ(expected, data);
}
