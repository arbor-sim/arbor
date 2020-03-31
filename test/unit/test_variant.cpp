#include <tuple>

#include <arbor/util/variant.hpp>
#include "util/meta.hpp"

#include "../gtest.h"
#include "common.hpp"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

using namespace arb::util;
using testing::nocopy;
using testing::nomove;

TEST(variant, in_place_index_ctor) {
    // Equal variant alternatives okay?
    {
        variant<int> v0{in_place_index<0>(), 3};
        ASSERT_EQ(0u, v0.index());
    }
    {
        variant<int, int> v0{in_place_index<0>(), 3};
        ASSERT_EQ(0u, v0.index());

        variant<int, int> v1{in_place_index<1>(), 3};
        ASSERT_EQ(1u, v1.index());
    }
    {
        variant<int, int, int> v0{in_place_index<0>(), 3};
        ASSERT_EQ(0u, v0.index());

        variant<int, int, int> v1{in_place_index<1>(), 3};
        ASSERT_EQ(1u, v1.index());

        variant<int, int, int> v2{in_place_index<2>(), 3};
        ASSERT_EQ(2u, v2.index());
    }

    // Check move- and copy- only types work.
    {
        struct foo { explicit foo(int, double) {} };
        nocopy<foo>::reset_counts();
        nomove<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v0(in_place_index<0>(), 1, 3.2);
        ASSERT_EQ(0u, v0.index());
        EXPECT_EQ(0, nocopy<foo>::move_ctor_count); // (should have constructed in-place)
        EXPECT_EQ(0, nocopy<foo>::move_assign_count);
        nocopy<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v0bis(in_place_index<0>(), nocopy<foo>(1, 3.2));
        ASSERT_EQ(0u, v0.index());
        EXPECT_EQ(1, nocopy<foo>::move_ctor_count); // (should have move-constructed)
        EXPECT_EQ(0, nocopy<foo>::move_assign_count); // (should have constructed in-place)
        nocopy<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v1(in_place_index<1>(), 1, 3.2);
        ASSERT_EQ(1u, v1.index());
        EXPECT_EQ(0, nomove<foo>::copy_ctor_count); // (should have constructed in-place)
        EXPECT_EQ(0, nomove<foo>::copy_assign_count);
        nomove<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v1bis(in_place_index<1>(), nomove<foo>(1, 3.2));
        ASSERT_EQ(1u, v1bis.index());
        EXPECT_EQ(1, nomove<foo>::copy_ctor_count); // (should have copy-constructed)
        EXPECT_EQ(0, nomove<foo>::copy_assign_count);
        nomove<foo>::reset_counts();
    }
}

TEST(variant, in_place_type_ctor) {
    {
        variant<int> v0{in_place_type<int>(), 3};
        ASSERT_EQ(0u, v0.index());
    }
    {
        variant<int, double> v0{in_place_type<int>(), 3};
        ASSERT_EQ(0u, v0.index());

        variant<int, double> v1{in_place_type<double>(), 3};
        ASSERT_EQ(1u, v1.index());
    }
    // Check move- and copy- only types for in_place_type too.
    {
        struct foo { explicit foo(int, double) {} };
        nocopy<foo>::reset_counts();
        nomove<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v0(in_place_type<nocopy<foo>>(), 1, 3.2);
        ASSERT_EQ(0u, v0.index());
        EXPECT_EQ(0, nocopy<foo>::move_ctor_count); // (should have constructed in-place)
        EXPECT_EQ(0, nocopy<foo>::move_assign_count);
        nocopy<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v0bis(in_place_type<nocopy<foo>>(), nocopy<foo>(1, 3.2));
        ASSERT_EQ(0u, v0.index());
        EXPECT_EQ(1, nocopy<foo>::move_ctor_count); // (should have move-constructed)
        EXPECT_EQ(0, nocopy<foo>::move_assign_count); // (should have constructed in-place)
        nocopy<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v1(in_place_type<nomove<foo>>(), 1, 3.2);
        ASSERT_EQ(1u, v1.index());
        EXPECT_EQ(0, nomove<foo>::copy_ctor_count); // (should have constructed in-place)
        EXPECT_EQ(0, nomove<foo>::copy_assign_count);
        nomove<foo>::reset_counts();

        variant<nocopy<foo>, nomove<foo>> v1bis(in_place_type<nomove<foo>>(), nomove<foo>(1, 3.2));
        ASSERT_EQ(1u, v1bis.index());
        EXPECT_EQ(1, nomove<foo>::copy_ctor_count); // (should have copy-constructed)
        EXPECT_EQ(0, nomove<foo>::copy_assign_count);
        nomove<foo>::reset_counts();
    }
}

TEST(variant, converting_ctor) {
    struct Z {};
    struct X { X() {} X(Z) {} };
    struct Y {};

    // Expect resolution via overload set of one-argument constructors.
    {
        using var_xy = variant<X, Y>;
        var_xy v0(X{});
        ASSERT_EQ(0u, v0.index());

        var_xy v1(Y{});
        ASSERT_EQ(1u, v1.index());

        var_xy v0bis(Z{});
        ASSERT_EQ(0u, v0bis.index());
    }
    {
        using var_xyz = variant<X, Y, Z>;
        var_xyz v0(X{});
        ASSERT_EQ(0u, v0.index());

        var_xyz v1(Y{});
        ASSERT_EQ(1u, v1.index());

        var_xyz v2(Z{});
        ASSERT_EQ(2u, v2.index());
    }

    // A bool alternative should only accept (cvref qualified) bool.
    {
        using bool_or_ptr = variant<bool, void*>;
        bool_or_ptr v0(false);
        ASSERT_EQ(0u, v0.index());

        bool_or_ptr v1(nullptr);
        ASSERT_EQ(1u, v1.index());
    }
}

TEST(variant, get) {
    struct X {};

    {
        variant<int, double, X> v(2.3);

        EXPECT_THROW(get<0>(v), bad_variant_access);
        EXPECT_EQ(2.3, get<1>(v));

        EXPECT_THROW(get<int>(v), bad_variant_access);
        EXPECT_EQ(2.3, get<double>(v));
    }
    {
        variant<nocopy<double>> v(3.1);
        auto x = get<0>(std::move(v));
        // nocopy will zero value on move
        EXPECT_EQ(3.1, x.value);
        EXPECT_EQ(0.0, get<0>(v).value);
    }
    {
        // should be able to modify in-place
        variant<double> v(3.1);
        get<0>(v) = 4.2;
        EXPECT_EQ(4.2, get<0>(v));
    }
}

TEST(variant, get_if) {
    struct X {};

    {
        variant<int, double, X> v(2.3);

        EXPECT_EQ(nullptr, get_if<0>(v));
        ASSERT_NE(nullptr, get_if<1>(v));
        EXPECT_EQ(2.3, *get_if<1>(v));

        EXPECT_EQ(nullptr, get_if<int>(v));
        ASSERT_NE(nullptr, get_if<double>(v));
        EXPECT_EQ(2.3, *get_if<double>(v));
    }
    {
        // should be able to modify in-place
        variant<double> v(3.1);
        ASSERT_NE(nullptr, get_if<0>(v));
        *get_if<0>(v) = 4.2;
        EXPECT_EQ(4.2, get<0>(v));
    }
}

TEST(variant, visit) {
    struct X {};

    // void case
    struct visitor {
        int* result = nullptr;
        visitor(int& r): result(&r) {}

        void operator()(int) { *result = 10; }
        void operator()(double) { *result = 11; }
        void operator()(X) { *result = 12; }
    };

    variant<int, double, X> v0(2);
    variant<int, double, X> v1(3.1);
    variant<int, double, X> v2(X{});

    int r;
    auto hello = visitor(r);

    visit<void>(hello, v0);
    EXPECT_EQ(10, r);

    visit<void>(hello, v1);
    EXPECT_EQ(11, r);

    visit<void>(hello, v2);
    EXPECT_EQ(12, r);
}

TEST(variant, visit_deduce_return) {
    struct X {};

    struct visitor {
        char operator()(int) { return 'i'; }
        char operator()(double) { return 'd'; }
        char operator()(X) { return 'X'; }
    } hello;

    using variant_idX = variant<int, double, X>;

    EXPECT_EQ('i', visit(hello, variant_idX(1)));
    EXPECT_EQ('d', visit(hello, variant_idX(1.1)));
    EXPECT_EQ('X', visit(hello, variant_idX(X{})));
}

TEST(variant, valueless) {
    struct X {
        X() {}
        X(const X&) { throw "nope"; }
    };

    variant<X, int> vx;
    variant<X, int> vi(3);

    ASSERT_EQ(0u, vx.index());
    ASSERT_EQ(1u, vi.index());
    try {
        vi = vx;
    }
    catch (...) {
    }
    EXPECT_TRUE(vi.valueless_by_exception());
    EXPECT_EQ(std::size_t(-1), vi.index());
}

TEST(variant, hash) {
    // Just ensure we find std::hash specializations.

    std::hash<variant<>> h0;
    EXPECT_TRUE((std::is_same<std::size_t, decltype(h0(std::declval<variant<>>()))>::value));

    std::hash<variant<int, double>> h2;
    EXPECT_TRUE((std::is_same<std::size_t, decltype(h2(std::declval<variant<int, double>>()))>::value));
}

namespace {
struct counts_swap {
    static unsigned n_swap;
    friend void swap(counts_swap&, counts_swap&) { ++counts_swap::n_swap; }
};
unsigned counts_swap::n_swap = 0;
}

TEST(variant, swap) {
    struct X {
        X() {}
        X& operator=(const X&) { throw "nope"; }
    };
    using vidX = variant<int, double, X>;

    auto valueless = []() {
        vidX v{X{}};
        try { v = v; } catch (...) {};
        return v;
    };

    {
        vidX a(valueless()), b(valueless());
        ASSERT_TRUE(a.valueless_by_exception());
        ASSERT_TRUE(b.valueless_by_exception());
        std::swap(a, b);
        EXPECT_TRUE(a.valueless_by_exception());
        EXPECT_TRUE(b.valueless_by_exception());
    };

    {
        vidX a(valueless()), b(3.2);
        ASSERT_TRUE(a.valueless_by_exception());
        ASSERT_EQ(1u, b.index());

        std::swap(a, b);
        EXPECT_TRUE(b.valueless_by_exception());
        EXPECT_EQ(1u, a.index());
        ASSERT_NE(nullptr, get_if<1>(a));
        EXPECT_EQ(3.2, get<1>(a));
    }

    {
        vidX a(1.2), b(3);
        std::swap(a, b);

        ASSERT_EQ(0u, a.index());
        EXPECT_EQ(3, get<int>(a));

        ASSERT_EQ(1u, b.index());
        EXPECT_EQ(1.2, get<double>(b));
    }

    {
        variant<counts_swap> y0, y1;
        ASSERT_EQ(0u, counts_swap::n_swap);

        std::swap(y0, y1);
        EXPECT_EQ(1u, counts_swap::n_swap);
    }
}

// Test generic accessors for pair, tuple.

TEST(variant, get_pair_tuple) {
    {
        using pair_ni_nd = std::pair<nocopy<int>, nocopy<double>>;

        nocopy<int>::reset_counts();
        nocopy<double>::reset_counts();

        auto f = first(pair_ni_nd{2, 3.4});
        EXPECT_EQ(2, f.value);
        EXPECT_EQ(1, nocopy<int>::move_ctor_count);

        auto s = second(pair_ni_nd{2, 3.4});
        EXPECT_EQ(3.4, s.value);
        EXPECT_EQ(1, nocopy<double>::move_ctor_count);

        nocopy<int>::reset_counts();
        nocopy<double>::reset_counts();

        auto g0 = ::arb::util::get<0>(pair_ni_nd{2, 3.4});
        EXPECT_EQ(2, g0.value);
        EXPECT_EQ(1, nocopy<int>::move_ctor_count);

        auto g1 = ::arb::util::get<1>(pair_ni_nd{2, 3.4});
        EXPECT_EQ(3.4, g1.value);
        EXPECT_EQ(1, nocopy<double>::move_ctor_count);
    }

    {
        struct X {};
        using tuple_ni_nd_nx = std::tuple<nocopy<int>, nocopy<double>, nocopy<X>>;

        nocopy<int>::reset_counts();
        nocopy<double>::reset_counts();
        nocopy<X>::reset_counts();

        auto f = first(tuple_ni_nd_nx{2, 3.4, X{}});
        EXPECT_EQ(2, f.value);
        EXPECT_EQ(1, nocopy<int>::move_ctor_count);

        auto s = second(tuple_ni_nd_nx{2, 3.4, X{}});
        EXPECT_EQ(3.4, s.value);
        EXPECT_EQ(1, nocopy<double>::move_ctor_count);

        nocopy<int>::reset_counts();
        nocopy<double>::reset_counts();

        auto g0 = ::arb::util::get<0>(tuple_ni_nd_nx{2, 3.4, X{}});
        EXPECT_EQ(2, g0.value);
        EXPECT_EQ(1, nocopy<int>::move_ctor_count);

        auto g1 = ::arb::util::get<1>(tuple_ni_nd_nx{2, 3.4, X{}});
        EXPECT_EQ(3.4, g1.value);
        EXPECT_EQ(1, nocopy<double>::move_ctor_count);

        auto g2 = ::arb::util::get<2>(tuple_ni_nd_nx{2, 3.4, X{}});
        (void)g2;
        EXPECT_EQ(1, nocopy<X>::move_ctor_count);
    }
}
