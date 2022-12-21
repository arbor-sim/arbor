#include <gtest/gtest.h>

#include <arbor/util/uninitialized.hpp>

#include "common.hpp"

using namespace arb::util;

namespace {
    struct count_ops {
        count_ops() {}
        count_ops(const count_ops& n) { ++copy_ctor_count; }
        count_ops(count_ops&& n) { ++move_ctor_count; }

        count_ops& operator=(const count_ops& n) { ++copy_assign_count; return *this; }
        count_ops& operator=(count_ops&& n) { ++move_assign_count; return *this; }

        static int copy_ctor_count, copy_assign_count;
        static int move_ctor_count, move_assign_count;
        static void reset_counts() {
            copy_ctor_count = copy_assign_count = 0;
            move_ctor_count = move_assign_count = 0;
        }
    };

    int count_ops::copy_ctor_count = 0;
    int count_ops::copy_assign_count = 0;
    int count_ops::move_ctor_count = 0;
    int count_ops::move_assign_count = 0;
}

TEST(uninitialized, ctor) {
    count_ops::reset_counts();

    uninitialized<count_ops> ua;
    ua.construct(count_ops{});

    count_ops b;
    ua.construct(b);

    EXPECT_EQ(1, count_ops::copy_ctor_count);
    EXPECT_EQ(0, count_ops::copy_assign_count);
    EXPECT_EQ(1, count_ops::move_ctor_count);
    EXPECT_EQ(0, count_ops::move_assign_count);

    ua.ref() = count_ops{};
    ua.ref() = b;

    EXPECT_EQ(1, count_ops::copy_ctor_count);
    EXPECT_EQ(1, count_ops::copy_assign_count);
    EXPECT_EQ(1, count_ops::move_ctor_count);
    EXPECT_EQ(1, count_ops::move_assign_count);
}

TEST(uninitialized, ctor_nocopy) {
    using nocopy = testing::nocopy<int>;
    nocopy::reset_counts();

    uninitialized<nocopy> ua;
    ua.construct(nocopy{});

    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(0, nocopy::move_assign_count);

    ua.ref() = nocopy{};

    EXPECT_EQ(1, nocopy::move_ctor_count);
    EXPECT_EQ(1, nocopy::move_assign_count);
}

TEST(uninitialized, ctor_nomove) {
    using nomove = testing::nomove<int>;
    nomove::reset_counts();

    uninitialized<nomove> ua;
    ua.construct(nomove{}); // check against rvalue

    nomove b;
    ua.construct(b); // check against non-const lvalue

    const nomove c;
    ua.construct(c); // check against const lvalue

    EXPECT_EQ(3, nomove::copy_ctor_count);
    EXPECT_EQ(0, nomove::copy_assign_count);

    nomove a;
    ua.ref() = a;

    EXPECT_EQ(3, nomove::copy_ctor_count);
    EXPECT_EQ(1, nomove::copy_assign_count);
}

TEST(uninitialized, void) {
    uninitialized<void> a, b;
    a = b;

    EXPECT_EQ(typeid(a.ref()), typeid(void));
}

TEST(uninitialized, ref) {
    uninitialized<int&> x, y;
    int a;

    x.construct(a);
    y = x;

    x.ref() = 2;
    EXPECT_EQ(2, a);

    y.ref() = 3;
    EXPECT_EQ(3, a);
    EXPECT_EQ(3, x.cref());

    EXPECT_EQ(&a, x.ptr());
    EXPECT_EQ((const int *)&a, x.cptr());
}
