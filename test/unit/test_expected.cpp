#include <gtest/gtest.h>

#include <utility>
#include <vector>
#include <arbor/util/expected.hpp>

#include "common.hpp"

using namespace arb::util;
using std::in_place;

TEST(expected, ctors) {
    // Test constructors, verify against bool conversion and
    // access via value() and error().

    struct int3 {
        int v = 3;
        int3() = default;
        int3(int a, int b, int c): v(a+b+c) {}
    };

    {
        // Default construction.

        expected<int3, int3> x;
        EXPECT_TRUE(x);
        EXPECT_EQ(3, x.value().v);
    }
    {
        // Default void construction.

        expected<void, int3> x;
        EXPECT_TRUE(x);
    }
    {
        // In-place construction.

        expected<int3, int3> x(in_place, 1, 2, 3);
        EXPECT_TRUE(x);
        EXPECT_EQ(6, x.value().v);
    }
    {
        // From-value construction.

        int3 v;
        v.v = 19;
        expected<int3, int3> x(v);
        EXPECT_TRUE(x);
        EXPECT_EQ(19, x.value().v);
    }
    {
        // From-unexpected construction.

        int3 v;
        v.v = 19;
        expected<int3, int3> x{unexpected(v)};
        EXPECT_FALSE(x);
        EXPECT_EQ(19, x.error().v);
        EXPECT_THROW(x.value(), bad_expected_access<int3>);
    }
    {
        // From-unexpected void construction.

        int3 v;
        v.v = 19;
        expected<void, int3> x{unexpected(v)};
        EXPECT_FALSE(x);
        EXPECT_EQ(19, x.error().v);
    }
    {
        // In-place unexpected construction.

        expected<int3, int3> x(unexpect, 1, 2, 3);
        EXPECT_FALSE(x);
        EXPECT_EQ(6, x.error().v);
        EXPECT_THROW(x.value(), bad_expected_access<int3>);
    }
    {
        // In-place void unexpected construction.

        expected<void, int3> x(unexpect, 1, 2, 3);
        EXPECT_FALSE(x);
        EXPECT_EQ(6, x.error().v);
    }
    {
        // Conversion from other expect.

        struct X {};
        struct Y {};
        struct Z {
            int v = 0;
            Z(const X&): v(1) {}
            Z(const Y&): v(2) {}
            Z(X&&): v(-1) {}
            Z(Y&&): v(-2) {}
        };

        expected<X, Y> x;
        expected<Z, Z> y(x);
        EXPECT_TRUE(y);
        EXPECT_EQ(1, y.value().v);

        expected<Z, Z> my(std::move(x));
        EXPECT_TRUE(my);
        EXPECT_EQ(-1, my.value().v);

        expected<X, Y> xu(unexpect);
        expected<Z, Z> yu(xu);
        EXPECT_FALSE(yu);
        EXPECT_EQ(2, yu.error().v);

        expected<Z, Z> myu(std::move(xu));
        EXPECT_FALSE(myu);
        EXPECT_EQ(-2, myu.error().v);
    }
}

TEST(expected, assignment) {
    {
        expected<int, int> x(10), y(12), z(unexpect, 20);

        EXPECT_EQ(12, (x=y).value());
        EXPECT_EQ(20, (x=z).error());

        expected<void, int> u, v, w(unexpect, 30);

        EXPECT_TRUE((u=v).has_value());
        EXPECT_EQ(30, (u=w).error());
    }

    {
        struct X {
            X(): v(0) {}
            X(const int& a): v(10*a) {}
            X(int&& a): v(20*a) {}
            int v;
        };

        expected<X, int> y;
        EXPECT_EQ(20, (y=1).value().v);
        int a = 3;
        EXPECT_EQ(30, (y=a).value().v);

        expected<int, X> z;
        EXPECT_EQ(20, (z=unexpected(1)).error().v);
        unexpected<int> b(3);
        EXPECT_EQ(30, (z=b).error().v);

        expected<void, X> v;
        EXPECT_EQ(20, (v=unexpected(1)).error().v);
        EXPECT_EQ(30, (v=b).error().v);
    }
}

TEST(expected, emplace) {
    // Check we're forwarding properly...
    struct X {
        X(): v(0) {}
        X(const int& a, int b): v(10*a + b) {}
        X(int&& a, int b): v(20*a + b) {}
        int v;
    };

    expected<X, bool> ex;
    EXPECT_TRUE(ex);
    EXPECT_EQ(0, ex.value().v);

    int i = 3, j = 4;
    ex.emplace(i, j);
    EXPECT_TRUE(ex);
    EXPECT_EQ(34, ex.value().v);
    ex.emplace(3, j);
    EXPECT_TRUE(ex);
    EXPECT_EQ(64, ex.value().v);

    // Should also work if ex was in error state.
    expected<X, bool> ux(unexpect);
    EXPECT_FALSE(ux);
    ux.emplace(4, 1);
    EXPECT_TRUE(ux);
    EXPECT_EQ(81, ux.value().v);
}

TEST(expected, equality) {
    {
        // non-void value expected comparisons:

        expected<int, int> ex1(1), ux1(unexpect, 1), ex2(2), ux2(unexpect, 2);
        expected<int, int> x(ex1);

        EXPECT_TRUE(x==ex1);
        EXPECT_TRUE(ex1==x);
        EXPECT_FALSE(x!=ex1);
        EXPECT_FALSE(ex1!=x);

        EXPECT_FALSE(x==ex2);
        EXPECT_FALSE(ex2==x);
        EXPECT_TRUE(x!=ex2);
        EXPECT_TRUE(ex2!=x);

        EXPECT_FALSE(x==ux1);
        EXPECT_FALSE(ux1==x);
        EXPECT_TRUE(x!=ux1);
        EXPECT_TRUE(ux1!=x);

        EXPECT_FALSE(ux1==ux2);
        EXPECT_FALSE(ux2==ux1);
        EXPECT_TRUE(ux1!=ux2);
        EXPECT_TRUE(ux2!=ux1);
    }
    {
        // non-void comparison against values and unexpected.

        expected<int, int> x(10);

        EXPECT_TRUE(x==10);
        EXPECT_TRUE(10==x);
        EXPECT_FALSE(x!=10);
        EXPECT_FALSE(10!=x);

        EXPECT_FALSE(x==unexpected(10));
        EXPECT_FALSE(unexpected(10)==x);
        EXPECT_TRUE(x!=unexpected(10));
        EXPECT_TRUE(unexpected(10)!=x);

        x = unexpected(10);

        EXPECT_FALSE(x==10);
        EXPECT_FALSE(10==x);
        EXPECT_TRUE(x!=10);
        EXPECT_TRUE(10!=x);

        EXPECT_TRUE(x==unexpected(10));
        EXPECT_TRUE(unexpected(10)==x);
        EXPECT_FALSE(x!=unexpected(10));
        EXPECT_FALSE(unexpected(10)!=x);
    }
    {
        // void value expected comparisons:

        expected<void, int> ev, uv1(unexpect, 1), uv2(unexpect, 2);
        expected<void, int> x(ev);

        EXPECT_TRUE(x==ev);
        EXPECT_TRUE(ev==x);
        EXPECT_FALSE(x!=ev);
        EXPECT_FALSE(ev!=x);

        EXPECT_FALSE(x==uv1);
        EXPECT_FALSE(uv1==x);
        EXPECT_TRUE(x!=uv1);
        EXPECT_TRUE(uv1!=x);

        EXPECT_FALSE(uv1==uv2);
        EXPECT_FALSE(uv2==uv1);
        EXPECT_TRUE(uv1!=uv2);
        EXPECT_TRUE(uv2!=uv1);
    }
    {
        // void value but difference unexpected types:

        expected<void, int> uvi(unexpect);
        expected<void, double> uvd(unexpect, 3.);

        EXPECT_FALSE(uvi==uvd);
        EXPECT_FALSE(uvd==uvi);

        uvi = expected<void, int>();
        ASSERT_TRUE(uvi);
        EXPECT_FALSE(uvi==uvd);
        EXPECT_FALSE(uvd==uvi);

        uvd = expected<void, double>();
        ASSERT_TRUE(uvd);
        EXPECT_TRUE(uvi==uvd);
        EXPECT_TRUE(uvd==uvi);
    }
    {
        // void comparison against unexpected.

        expected<void, int> x;

        EXPECT_TRUE(x);
        EXPECT_FALSE(x==unexpected(10));
        EXPECT_FALSE(unexpected(10)==x);
        EXPECT_TRUE(x!=unexpected(10));
        EXPECT_TRUE(unexpected(10)!=x);

        x = unexpected<int>(10);

        EXPECT_FALSE(x);
        EXPECT_TRUE(x==unexpected(10));
        EXPECT_TRUE(unexpected(10)==x);
        EXPECT_FALSE(x!=unexpected(10));
        EXPECT_FALSE(unexpected(10)!=x);
    }
}

TEST(expected, value_or) {
    expected<double, char> a(2.0), b(unexpect, 'x');
    EXPECT_EQ(2.0, a.value_or(1));
    EXPECT_EQ(1.0, b.value_or(1));
}

namespace {
struct Xswap {
    explicit Xswap(int val, int& r): val(val), n_swap_ptr(&r) {}
    int val;
    int* n_swap_ptr;
    void swap(Xswap& other) {
        ++*n_swap_ptr;
        std::swap(val, other.val);
    }
    friend void swap(Xswap& x1, Xswap& x2) noexcept { x1.swap(x2); }
};

struct swap_can_throw {
    friend void swap(swap_can_throw& s1, swap_can_throw& s2) noexcept(false) {}
};
}

TEST(expected, swap) {
    int swaps = 0;
    expected<Xswap, int> x1(in_place, -1, swaps), x2(in_place, -2, swaps);

    using std::swap;
    swap(x1, x2);
    EXPECT_EQ(-2, x1->val);
    EXPECT_EQ(-1, x2->val);
    EXPECT_EQ(1, swaps);

    swaps = 0;
    expected<Xswap, int> x3(unexpect, 4);
    swap(x1, x3);
    EXPECT_EQ(4, x1.error());
    EXPECT_EQ(-2, x3->val);
    EXPECT_EQ(0, swaps); // Xswap is moved, not swapped.

    swaps = 0;
    unexpected<Xswap> u1(in_place, -1, swaps), u2(in_place, -2, swaps);
    swap(u1, u2);
    EXPECT_EQ(-2, u1.value().val);
    EXPECT_EQ(-1, u2.value().val);
    EXPECT_EQ(1, swaps);

    EXPECT_TRUE(std::is_nothrow_swappable<unexpected<Xswap>>::value);
    EXPECT_FALSE(std::is_nothrow_swappable<unexpected<swap_can_throw>>::value);
}
