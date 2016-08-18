#include "gtest.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <list>
#include <numeric>
#include <type_traits>

#ifdef WITH_TBB
#include <tbb/tbb_stddef.h>
#endif

#include <util/range.hpp>

using namespace nest::mc;

TEST(range, list_iterator) {
    std::list<int> l = { 2, 4, 6, 8, 10 };

    auto s = util::make_range(l.begin(), l.end());

    EXPECT_EQ(s.left, l.begin());
    EXPECT_EQ(s.right, l.end());

    EXPECT_EQ(s.begin(), l.begin());
    EXPECT_EQ(s.end(), l.end());

    EXPECT_EQ(s.size(), l.size());
    EXPECT_EQ(s.front(), *l.begin());
    EXPECT_EQ(s.back(), *std::prev(l.end()));

    int check = std::accumulate(l.begin(), l.end(), 0);
    int sum = 0;
    for (auto i: s) {
        sum += i;
    }

    EXPECT_EQ(check, sum);
}

TEST(range, pointer) {
    int xs[] = { 10, 11, 12, 13, 14, 15, 16 };
    int l = 2;
    int r = 5;

    util::range<int *> s(&xs[l], &xs[r]);

    EXPECT_EQ(3u, s.size());

    EXPECT_EQ(xs[l], *s.left);
    EXPECT_EQ(xs[l], *s.begin());
    EXPECT_EQ(xs[l], s[0]);
    EXPECT_EQ(xs[l], s.front());

    EXPECT_EQ(xs[r], *s.right);
    EXPECT_EQ(xs[r], *s.end());
    EXPECT_THROW(s.at(r-l), std::out_of_range);

    EXPECT_EQ(r-l, std::distance(s.begin(), s.end()));

    EXPECT_TRUE(std::equal(s.begin(), s.end(), &xs[l]));
}

TEST(range, input_iterator) {
    int nums[] = { 10, 9, 8, 7, 6 };
    std::istringstream sin("10 9 8 7 6");
    auto s = util::make_range(std::istream_iterator<int>(sin), std::istream_iterator<int>());

    EXPECT_TRUE(std::equal(s.begin(), s.end(), &nums[0]));
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

TEST(range, sentinel) {
    const char *cstr = "hello world";
    std::string s;

    auto cstr_range = util::make_range(cstr, null_terminated);
    for (auto i=cstr_range.begin(); i!=cstr_range.end(); ++i) {
        s += *i;
    }

    EXPECT_EQ(s, std::string(cstr));

    s.clear();
    for (auto c: canonical_view(cstr_range)) {
        s += c;
    }

    EXPECT_EQ(s, std::string(cstr));
}

#ifdef WITH_TBB

TEST(range, tbb_split) {
    constexpr std::size_t N = 20;
    int xs[N];

    for (unsigned i = 0; i<N; ++i) {
        xs[i] = i;
    }

    auto s = util::make_range(&xs[0], &xs[0]+N);

    while (s.size()>1) {
        auto ssize = s.size();
        auto r = decltype(s){s, tbb::split{}};
        EXPECT_GT(r.size(), 0u);
        EXPECT_GT(s.size(), 0u);
        EXPECT_EQ(ssize, r.size()+s.size());
        EXPECT_EQ(s.end(), r.begin());

        EXPECT_TRUE(r.size()>1 || !r.is_divisible());
        EXPECT_TRUE(s.size()>1 || !s.is_divisible());
    }

    for (unsigned i = 1; i<N-1; ++i) {
        s = util::make_range(&xs[0], &xs[0]+N);
        // expect exact splitting by proportion in this instance

        auto r = decltype(s){s, tbb::proportional_split{i, N-i}};
        EXPECT_EQ(&xs[0], s.left);
        EXPECT_EQ(&xs[0]+i, s.right);
        EXPECT_EQ(&xs[0]+i, r.left);
        EXPECT_EQ(&xs[0]+N, r.right);
    }
}

TEST(range, tbb_no_split) {
    std::istringstream sin("10 9 8 7 6");
    auto s = util::make_range(std::istream_iterator<int>(sin), std::istream_iterator<int>());

    EXPECT_FALSE(decltype(s)::is_splittable_in_proportion());
    EXPECT_FALSE(s.is_divisible());
}

#endif
