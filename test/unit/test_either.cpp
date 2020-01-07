#include <typeinfo>
#include <array>
#include <algorithm>

#include <arbor/util/either.hpp>

#include "../gtest.h"

// TODO: coverage!

using namespace arb::util;

TEST(either, basic) {
    either<int, std::string> e0(17);

    EXPECT_TRUE(e0);
    EXPECT_EQ(17, e0.get<0>());
    EXPECT_EQ(e0.unsafe_get<0>(), e0.get<0>());
    EXPECT_EQ(e0.unsafe_get<0>(), e0.first());
    EXPECT_THROW(e0.get<1>(), either_invalid_access);
    either<int, std::string> e1("seventeen");

    EXPECT_FALSE(e1);
    EXPECT_EQ("seventeen", e1.get<1>());
    EXPECT_EQ(e1.unsafe_get<1>(), e1.get<1>());
    EXPECT_EQ(e1.unsafe_get<1>(), e1.second());
    EXPECT_THROW(e1.get<0>(), either_invalid_access);

    e0 = e1;
    EXPECT_EQ("seventeen", e0.get<1>());
    EXPECT_THROW(e0.get<0>(), either_invalid_access);

    e0 = 19;
    EXPECT_EQ(19, e0.get<0>());
}

struct no_copy {
    int value;

    no_copy(): value(23) {}
    explicit no_copy(int v): value(v) {}
    no_copy(const no_copy&) = delete;
    no_copy(no_copy&&) = default;

    no_copy& operator=(const no_copy&) = delete;
    no_copy& operator=(no_copy&&) = default;
};

TEST(either, no_copy) {
    either<no_copy, std::string> e0(no_copy{17});

    EXPECT_TRUE(e0);

    either<no_copy, std::string> e1(std::move(e0));

    EXPECT_TRUE(e1);

    either<no_copy, std::string> e2;
    EXPECT_TRUE(e2);
    EXPECT_EQ(23, e2.get<0>().value);

    e2 = std::move(e1);
    EXPECT_TRUE(e2);
    EXPECT_EQ(17, e2.get<0>().value);
}
