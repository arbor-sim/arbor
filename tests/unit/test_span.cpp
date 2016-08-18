#include "gtest.h"

#include <algorithm>
#include <iterator>
#include <list>
#include <numeric>
#include <type_traits>

#include <util/span.hpp>

using namespace nest::mc;

TEST(span, int_access) {
    using span = util::span<int>;

    int n = 97;
    int a = 3;
    int b = a+n;

    span s(a, b);
    EXPECT_EQ(s.left, a);
    EXPECT_EQ(s.right, b);

    EXPECT_EQ(s.size(), std::size_t(n));

    EXPECT_EQ(s.front(), a);
    EXPECT_EQ(s.back(), b-1);

    EXPECT_EQ(s[0], a);
    EXPECT_EQ(s[1], a+1);
    EXPECT_EQ(s[n-1], b-1);

    EXPECT_NO_THROW(s.at(0));
    EXPECT_NO_THROW(s.at(n-1));
    EXPECT_THROW(s.at(n), std::out_of_range);
    EXPECT_THROW(s.at(n+1), std::out_of_range);
    EXPECT_THROW(s.at(-1), std::out_of_range);
}

TEST(span, int_iterators) {
    using span = util::span<int>;

    int n = 97;
    int a = 3;
    int b = a+n;

    span s(a, b);

    EXPECT_TRUE(util::is_iterator<span::iterator>::value);
    EXPECT_TRUE(util::is_random_access_iterator<span::iterator>::value);

    EXPECT_EQ(n, std::distance(s.begin(), s.end()));
    EXPECT_EQ(n, std::distance(s.cbegin(), s.cend()));

    int sum = 0;
    for (auto i: span(a, b)) {
        sum += i;
    }
    EXPECT_EQ(sum, (a+b-1)*(b-a)/2);
}

