#include "../gtest.h"

#include <iterator>
#include <type_traits>

#include "util/counter.hpp"

using namespace arb;

template <typename V>
class counter_test: public ::testing::Test {};

TYPED_TEST_CASE_P(counter_test);

TYPED_TEST_P(counter_test, value) {
    using int_type = TypeParam;
    using counter = util::counter<int_type>;

    counter c0;
    EXPECT_EQ(int_type{0}, *c0);

    counter c1{int_type{1}};
    counter c2{int_type{2}};

    EXPECT_EQ(int_type{1}, *c1);
    EXPECT_EQ(int_type{2}, *c2);

    c2 = c1;
    EXPECT_EQ(int_type{1}, *c2);

    c2 = int_type{2};
    EXPECT_EQ(int_type{2}, *c2);
}

TYPED_TEST_P(counter_test, compare) {
    using int_type = TypeParam;
    using counter = util::counter<int_type>;

    counter c1{int_type{1}};
    counter c2{int_type{2}};

    EXPECT_LT(c1, c2);
    EXPECT_LE(c1, c2);
    EXPECT_NE(c1, c2);
    EXPECT_GE(c2, c1);
    EXPECT_GT(c2, c1);

    counter c1bis{int_type{1}};

    EXPECT_LE(c1, c1bis);
    EXPECT_EQ(c1, c1bis);
    EXPECT_GE(c1, c1bis);
}

TYPED_TEST_P(counter_test, arithmetic) {
    using int_type = TypeParam;
    using counter = util::counter<int_type>;

    counter c1{int_type{1}};
    counter c2{int_type{10}};
    int_type nine{9};

    EXPECT_EQ(nine, c2-c1);
    EXPECT_EQ(c2, c1+nine);
    EXPECT_EQ(c2, nine+c1);
    EXPECT_EQ(c1, c2-nine);

    counter c3 = c1;
    counter c4 = (c3 += nine);

    EXPECT_EQ(c2, c3);
    EXPECT_EQ(c3, c4);

    c3 = c2;
    c4 = (c3 -= nine);

    EXPECT_EQ(c1, c3);
    EXPECT_EQ(c3, c4);

    c3 = c1;
    EXPECT_EQ(counter{2}, ++c3);
    EXPECT_EQ(counter{3}, ++c3);
    EXPECT_EQ(counter{2}, --c3);
    EXPECT_EQ(counter{1}, --c3);

    c3 = c1;
    EXPECT_EQ(counter{1}, c3++);
    EXPECT_EQ(counter{2}, c3++);
    EXPECT_EQ(counter{3}, c3--);
    EXPECT_EQ(counter{2}, c3--);

    EXPECT_EQ(int_type{10}, c2[0]);
    EXPECT_EQ(int_type{4},  c2[-6]);
    EXPECT_EQ(int_type{19}, c2[9]);
}

TYPED_TEST_P(counter_test, iterator_traits) {
    using int_type = TypeParam;
    using counter = util::counter<int_type>;
    using traits = std::iterator_traits<counter>;

    typename traits::reference r = *counter{int_type{3}};
    EXPECT_EQ(r, int_type{3});

    typename traits::difference_type d = counter{int_type{4}} - counter{int_type{7}};
    EXPECT_EQ(typename traits::difference_type(-3), d);

    EXPECT_TRUE((std::is_same<std::random_access_iterator_tag, typename traits::iterator_category>::value));
}

TYPED_TEST_P(counter_test, iterator_functions) {
    using int_type = TypeParam;
    using counter = util::counter<int_type>;

    counter c1{int_type{1}};
    counter c2{int_type{10}};

    EXPECT_EQ(int_type{9}, std::distance(c1,c2));
    counter c3{c1};
    std::advance(c3, int_type{9});
    EXPECT_EQ(c2, c3);

    EXPECT_EQ(counter{int_type{2}}, std::next(c1));
    EXPECT_EQ(counter{int_type{9}}, std::prev(c2));
}

REGISTER_TYPED_TEST_CASE_P(counter_test, value, compare, arithmetic, iterator_traits, iterator_functions);

using int_types = ::testing::Types<signed char, unsigned char, short, unsigned short, int, unsigned, std::size_t, std::ptrdiff_t>;
INSTANTIATE_TYPED_TEST_CASE_P(int_types, counter_test, int_types);
