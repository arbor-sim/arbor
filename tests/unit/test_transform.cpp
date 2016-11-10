#include "../gtest.h"

#include <cctype>
#include <forward_list>
#include <vector>

#include <util/range.hpp>
#include <util/transform.hpp>

using namespace nest::mc;

TEST(transform, transform_view) {
    std::forward_list<int> fl = {1, 4, 6, 8, 10 };
    std::vector<double> result;

    auto r = util::transform_view(fl, [](int i) { return i*i+0.5; });

    EXPECT_EQ(5u, util::size(r));
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

