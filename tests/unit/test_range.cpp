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

#include <util/counter.hpp>
#include <util/range.hpp>
#include <util/sentinel.hpp>

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

    auto sum2 = std::accumulate(s.begin(), s.end(), 0);
    EXPECT_EQ(check, sum2);
}

TEST(range, pointer) {
    int xs[] = { 10, 11, 12, 13, 14, 15, 16 };
    int l = 2;
    int r = 5;

    util::range<int *> s(&xs[l], &xs[r]);
    auto s_deduced = util::make_range(xs+l, xs+r);

    EXPECT_TRUE((std::is_same<decltype(s), decltype(s_deduced)>::value));
    EXPECT_EQ(s.left, s_deduced.left);
    EXPECT_EQ(s.right, s_deduced.right);

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

TEST(range, empty) {
    int xs[] = { 10, 11, 12, 13, 14, 15, 16 };
    auto l = 2;
    auto r = 5;

    auto empty_range_ll = util::make_range(&xs[l], &xs[l]);
    EXPECT_TRUE(empty_range_ll.empty());
    EXPECT_EQ(empty_range_ll.begin() == empty_range_ll.end(),
              empty_range_ll.empty());
    EXPECT_EQ(0u, empty_range_ll.size());


    auto empty_range_rr = util::make_range(&xs[r], &xs[r]);
    EXPECT_TRUE(empty_range_rr.empty());
    EXPECT_EQ(empty_range_rr.begin() == empty_range_rr.end(),
              empty_range_rr.empty());
    EXPECT_EQ(0u, empty_range_rr.size());
}

TEST(range, input_iterator) {
    int nums[] = { 10, 9, 8, 7, 6 };
    std::istringstream sin("10 9 8 7 6");
    auto s = util::make_range(std::istream_iterator<int>(sin), std::istream_iterator<int>());

    EXPECT_TRUE(std::equal(s.begin(), s.end(), &nums[0]));
}

TEST(range, const_iterator) {
    std::vector<int> xs = { 1, 2, 3, 4, 5 };
    auto r = util::make_range(xs.begin(), xs.end());
    EXPECT_TRUE((std::is_same<int&, decltype(r.front())>::value));

    const auto& xs_const = xs;
    auto r_const = util::make_range(xs_const.begin(), xs_const.end());
    EXPECT_TRUE((std::is_same<const int&, decltype(r_const.front())>::value));
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
    EXPECT_EQ(s.size(), cstr_range.size());

    s.clear();
    for (auto c: canonical_view(cstr_range)) {
        s += c;
    }

    EXPECT_EQ(s, std::string(cstr));

    const char *empty_cstr = "";
    auto empty_cstr_range = util::make_range(empty_cstr, null_terminated);
    EXPECT_TRUE(empty_cstr_range.empty());
    EXPECT_EQ(0u, empty_cstr_range.size());
}

template <typename V>
class counter_range: public ::testing::Test {};

TYPED_TEST_CASE_P(counter_range);

TYPED_TEST_P(counter_range, max_size) {
    using int_type = TypeParam;
    using unsigned_int_type = typename std::make_unsigned<int_type>::type;
    using counter = util::counter<int_type>;

    auto l = counter{int_type{1}};
    auto r = counter{int_type{10}};
    auto range = util::make_range(l, r);
    EXPECT_EQ(std::numeric_limits<unsigned_int_type>::max(),
              range.max_size());

}

TYPED_TEST_P(counter_range, extreme_size) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using unsigned_int_type = typename std::make_unsigned<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{std::numeric_limits<signed_int_type>::min()};
    auto r = counter{std::numeric_limits<signed_int_type>::max()};
    auto range = util::make_range(l, r);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(std::numeric_limits<unsigned_int_type>::max(), range.size());

}


TYPED_TEST_P(counter_range, size) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{signed_int_type{-3}};
    auto r = counter{signed_int_type{3}};
    auto range = util::make_range(l, r);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(6, range.size());

}

TYPED_TEST_P(counter_range, at) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto l = counter{signed_int_type{-3}};
    auto r = counter{signed_int_type{3}};
    auto range = util::make_range(l, r);
    EXPECT_EQ(range.front(), range.at(0));
    EXPECT_EQ(0, range.at(3));
    EXPECT_EQ(range.back(), range.at(range.size()-1));
    EXPECT_THROW(range.at(range.size()), std::out_of_range);
    EXPECT_THROW(range.at(30), std::out_of_range);
}

TYPED_TEST_P(counter_range, iteration) {
    using int_type = TypeParam;
    using signed_int_type = typename std::make_signed<int_type>::type;
    using counter = util::counter<signed_int_type>;

    auto j = signed_int_type{-3};
    auto l = counter{signed_int_type{j}};
    auto r = counter{signed_int_type{3}};

    for (auto i : util::make_range(l, r)) {
        EXPECT_EQ(j++, i);
    }
}

REGISTER_TYPED_TEST_CASE_P(counter_range, max_size, extreme_size, size, at,
                           iteration);

using int_types = ::testing::Types<signed char, unsigned char, short,
                                   unsigned short, int, unsigned, long,
                                   std::size_t, std::ptrdiff_t>;

using signed_int_types = ::testing::Types<signed char, short,
                                          int, long, std::ptrdiff_t>;

INSTANTIATE_TYPED_TEST_CASE_P(int_types, counter_range, int_types);

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
